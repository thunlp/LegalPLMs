import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler as lrs
import shutil
from timeit import default_timer as timer

from tools.eval_tool import valid, gen_time_str, output_value
from tools.init_tool import init_test_dataset, init_formatter
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


def checkpoint(filename, model, optimizer, trained_epoch, config, global_step, lr_scheduler):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step,
        "lr_scheduler": lr_scheduler.state_dict(),
    }

    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))


def train(parameters, config, gpu_list, do_test=False, local_rank=-1):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path):
        logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"] + 1
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]

    if do_test:
        init_formatter(config, ["test"])
        test_dataset = init_test_dataset(config)

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    grad_accumulate = config.getint("train", "grad_accumulate")

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.getint('train', 'warmup_steps'), num_training_steps=config.getint('train', 'training_steps'))
    #if "lr_scheduler" in parameters:
    #lr_scheduler.load_state_dict(parameters["lr_scheduler"])
    
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # exp_lr_scheduler.step(trained_epoch)

    fp16 = config.getboolean('train', 'fp16')
    if fp16:
        scaler = torch.cuda.amp.GradScaler()
    max_grad_norm = config.getfloat('train', 'max_grad_norm')
    valid_mode = config.get('train', 'valid_mode')
    if valid_mode != 'step' and valid_mode != 'batch':
        raise ValueError('The value of valid_mode is invalid.')
    print('valid_mode', valid_mode)
    if valid_mode == 'step':
        step_epoch = config.getint('train', 'step_epoch')
    print('step_epoch', step_epoch)
    logger.info("Training start....")

    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")

    total_len = len(dataset)
    more = ""
    if total_len < 10000:
        more = "\t"
    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num

        # exp_lr_scheduler.step(current_epoch)

        acc_result = None
        total_loss = 0

        output_info = ""
        step = -1
        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            if fp16:
                with autocast():
                    results = model(data, config, gpu_list, acc_result, "train")
            else:
                results = model(data, config, gpu_list, acc_result, "train")

            loss, acc_result = results["loss"], results["acc_result"]
            total_loss += float(loss)

            loss = loss / grad_accumulate
            if fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accumulate == 0:
                
                if max_grad_norm is not None and max_grad_norm > 0:
                    if fp16:
                        scaler.unscale_(optimizer)
                    if hasattr(optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        optimizer.clip_grad_norm(max_grad_norm)
                    elif hasattr(model, "clip_grad_norm_"):
                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                        model.clip_grad_norm_(max_grad_norm)
                    else:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_grad_norm
                        )

                if fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % output_time == 0 and local_rank <= 0:
                output_info = output_function(acc_result, config)

                delta_t = timer() - start_time

                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                             "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

            global_step += 1
            if (step + 1) % grad_accumulate == 0 and valid_mode == 'step' and int((step + 1) / grad_accumulate) % step_epoch == 0:
                if local_rank <= 0:
                    print()
                    checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config, global_step, lr_scheduler)
                    path = os.path.join(output_path, 'model_%d_%d' % (current_epoch, (step + 1) // grad_accumulate))
                    if local_rank < 0:
                        model.save_pretrained(path)
                    else:
                        model.module.save_pretrained(path)
                with torch.no_grad():
                    valid(model, parameters["valid_dataset"], current_epoch, config, gpu_list, output_function)

        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError


        if valid_mode != 'batch':
            continue

        if local_rank <= 0:
            output_info = output_function(acc_result, config)
            delta_t = timer() - start_time
            output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                        "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

        # if step == -1:
        #    logger.error("There is no data given to the model in this epoch, check your data.")
        #    raise NotImplementedError

        if local_rank <= 0:
            checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config, global_step, lr_scheduler)

        if current_epoch % test_time == 0:
            with torch.no_grad():
                valid(model, parameters["valid_dataset"], current_epoch, config, gpu_list, output_function)
                if do_test:
                    valid(model, test_dataset, current_epoch, config, gpu_list, output_function, mode="test")
