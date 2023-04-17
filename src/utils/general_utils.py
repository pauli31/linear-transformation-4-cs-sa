import datetime
import logging
import torch

logger = logging.getLogger(__name__)


def get_actual_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H.%M_%S.%f")

def print_time_info(time_sec, total_examples, label, per_examples=1000, print_fce=logger.info, file=None):
    formatted_time = format_time(time_sec)
    time_per_examples = time_sec / total_examples * per_examples
    print_fce(100 * '$$$$$$')
    print_fce("Times for:" + str(label))
    print_fce(f'Examples:{total_examples}')
    print_fce(f'Total time for {str(label)} format: {formatted_time}')
    print_fce(f'Total time for {str(label)} in sec: {time_sec}')
    print_fce('----')
    print_fce(f'Total time for {str(per_examples)} examples format: {format_time(time_per_examples)}')
    print_fce(f'Total time for {str(per_examples)} examples in sec: {time_per_examples}')
    print_fce('----')
    print_fce('Copy ')
    # label | per_examples | total_examples | formatted_time | time_sec | time_per_examples
    output = label + '\t' + str(per_examples) + '\t' + str(total_examples) + '\t' + str(formatted_time) + \
             '\t' + str(time_sec) + '\t' + str(time_per_examples)
    print_fce(output)
    # write results to disk
    if file is not None:
        file_write = file + "_" + label + ".txt"
        with open(file_write, 'a', encoding='utf-8') as f:
            f.write(output + "\n")

    print_fce(100 * '$$$$$$')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def generate_file_name(args):
    time = get_actual_time()
    epochs = args.epoch_num
    binary = args.binary
    batch_size = args.batch_size
    model_name = args.model_name
    full_mode = args.full_mode
    max_train_data = args.max_train_data
    if max_train_data > 1:
        max_train_data = int(max_train_data)

    dataset_name = args.dataset_name

    if args.eval is True:
        model_name = model_name[:30]

    num_iter = 1
    if full_mode is True:
        num_iter = epochs

    name_files = {}
    name_file = None

    for i in range(1, num_iter + 1):
        # if we are in full mode we change the epochs
        if full_mode is True:
            epochs = i

        name_file = model_name + "_" \
                    + dataset_name \
                    + "_BS-" + str(batch_size) \
                    + "_EC-" + str(epochs) \
                    + "_LR-%.7f" % (args.lr) \
                    + "_LEN-" + str(args.max_seq_len) \
                    + "_SCH-" + str(args.scheduler) \
                    + "_WR-" + str(args.warm_up_steps) \
                    + "_TRN-" + str(args.use_only_train_data) \
                    + "_MXT-" + str(max_train_data) \
                    + "_BIN-" + str(binary) \
                    + "_WD-%.5f" % args.weight_decay \
                    + "_F-" + str(full_mode)

        name_file += "_" + time
        name_file = name_file.replace('.', '-')
        name_files[i] = name_file

    if full_mode is False:
        name_files = name_file

    return name_files


def print_gpu_info():
    try:
        logger.info(f"GPU first device name: {torch.cuda.get_device_name(0)}")
        # t = torch.cuda.get_device_properties(0).total_memory
        # c = torch.cuda.memory_cached(0)
        # a = torch.cuda.memory_allocated(0)
        # f = c - a  # free inside cache
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        t = info.total
        f = info.free
        a = info.used
        logger.info(f'GPU 0 Memory total    : {int(t / (1024.0 * 1024.0))} MiB')
        logger.info(f'GPU 0 Memory free     : {int(f / (1024.0 * 1024.0))} MiB')
        logger.info(f'GPU 0 Memory used     : {int(a / (1024.0 * 1024.0))} MiB')

        logger.info(f"GPU first device name: {torch.cuda.get_device_name(1)}")
        # t = torch.cuda.get_device_properties(0).total_memory
        # c = torch.cuda.memory_cached(0)
        # a = torch.cuda.memory_allocated(0)
        # f = c - a  # free inside cache
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(1)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        t = info.total
        f = info.free
        a = info.used
        logger.info(f'GPU 0 Memory total    : {int(t / (1024.0 * 1024.0))} MiB')
        logger.info(f'GPU 0 Memory free     : {int(f / (1024.0 * 1024.0))} MiB')
        logger.info(f'GPU 0 Memory used     : {int(a / (1024.0 * 1024.0))} MiB')
        # print(f'cached   : {c/(1024.0*1024.0)}')
    except Exception as e:
        logger.info("Exception during: " + str(e))
