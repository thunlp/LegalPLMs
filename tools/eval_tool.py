import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from timeit import default_timer as timer

logger = logging.getLogger(__name__)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 10:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 18:
        s += " "
    s = s + str(step) + " "
    while len(s) < 30:
        s += " "
    s += str(time)
    while len(s) < 45:
        s += " "
    s += str(loss)
    while len(s) < 53:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if not (end is None):
        print(s, end=end)
    else:
        print(s)


def valid(model, dataset, epoch, config, gpu_list, output_function, mode="valid"):
    model.eval()
    local_rank = config.getint('distributed', 'local_rank')

    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""

    output_time = config.getint("output", "output_time")
    step = -1
    more = ""
    if total_len < 10000:
        more = "\t"

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, config, gpu_list, acc_result, "valid")

        loss, acc_result = results["loss"], results["acc_result"]
        total_loss += float(loss)
        cnt += 1

        if step % output_time == 0 and local_rank <= 0:
            delta_t = timer() - start_time

            output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    if config.getboolean("distributed", "use"):
        shape = len(acc_result) + 1
        mytensor = torch.FloatTensor([total_loss] + [acc_result[key] for key in acc_result]).to(gpu_list[local_rank])
        mylist = [torch.FloatTensor(shape).to(gpu_list[local_rank]) for i in range(config.getint('distributed', 'gpu_num'))]
        torch.distributed.all_gather(mylist, mytensor)#, 0)
        if local_rank == 0:
            mytensor = sum(mylist)
            total_loss = float(mytensor[0]) / config.getint('distributed', 'gpu_num')
            index = 1
            for key in acc_result:
                acc_result[key] = int(mytensor[index])
                index += 1
    if local_rank <= 0:
        delta_t = timer() - start_time
        output_info = output_function(acc_result, config)
        output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    model.train()
