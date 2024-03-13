import argparse
import random
import numpy as np
import torch

from guided_diffusion import dist_util
from guided_diffusion import logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.dataloader import IRSTD_Dataset, SIRST_Dataset, NUDT_Dataset
from guided_diffusion.resample import create_named_schedule_sampler


def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == 'IRSTD':
        dataset = IRSTD_Dataset(args, args.data_dir)
        test_dataset = IRSTD_Dataset(args, args.data_dir, mode='test')
        args.in_ch = 4
    elif args.data_name == 'SIRST':
        dataset = SIRST_Dataset(args, args.data_dir)
        test_dataset = SIRST_Dataset(args, args.data_dir, mode='test')
        args.in_ch = 4
    elif args.data_name == 'NUDT':
        dataset = NUDT_Dataset(args, args.data_dir)
        test_dataset = NUDT_Dataset(args, args.data_dir, mode='test')
        args.in_ch = 4

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    data = iter(dataloader)
 
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device=torch.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=args.diffusion_steps)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=dataloader,
        test_dataloader=test_dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name='IRSTD',
        data_dir="../data/NUDT-SIRST",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=200,
        save_interval=10000,
        resume_checkpoint='',  # '"../results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev="0",
        multi_gpu=None,  # "0,1,2"
        out_dir=r'../results/results_NUDT'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
