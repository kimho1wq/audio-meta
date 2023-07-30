# This is a sample Python script.

# TEST : Audio feature extraction (like a spotify)
import yaml
import os

from pathlib import Path

from assets.config.config import PipelineVersion
from src.database import DBCollector
from src.pipeline.gen import PipelineGenerator, PipelineSpecGenerator
from src.extraction.data import Extractor

  
def db_audio_gen(pipeline_version:PipelineVersion, process_type: str, config_dict: dict, num_workers: int):
    out_dir = Path(config_dict['db_audio_dir']).joinpath(
        pipeline_version.get_audio_db_fn()).joinpath(process_type)

    # input dir of each collector
    collector = {
        'audio_meta': Path(config_dict['track_list']),
        'audio_analysis': Path(config_dict['track_list']),
        'music_detection': None
    }

    def proc():
        DBCollector.COLLECTOR[process_type](
            in_dir=collector[process_type],
            out_dir=out_dir,
            audio_db_config=config_dict['audio_db_config'][pipeline_version.database_ver],
            num_workers=num_workers
        ).operate()

    if not os.path.exists(out_dir):
        proc()


def db_preprocessing(pipeline_version:PipelineVersion, process_type: str, config_dict: dict, num_workers: int):
    out_dir = Path(config_dict['db_preproc_dir']).joinpath(
        pipeline_version.get_preproc_fn()).joinpath(process_type)
    in_dir=Path(config_dict['db_audio_dir']).joinpath(
        pipeline_version.get_audio_db_fn()).joinpath(process_type)

    def proc():
        PipelineGenerator.GENERATOR[process_type](
            in_dir=in_dir,
            out_dir=out_dir,
            pre_proc_config=config_dict['preproc_config'][pipeline_version.pre_processing_ver],
            num_workers=num_workers
        ).generate()

    if not os.path.exists(out_dir):
        proc()


def db_handcrafted_feats_gen(pipeline_version:PipelineVersion, process_type: str, config_dict: dict, num_workers: int):
    out_dir = Path(config_dict['handcrafted_feats_dir']).joinpath(
        pipeline_version.get_transform_fn()).joinpath(process_type)
    in_dir=Path(config_dict['db_preproc_dir']).joinpath(
        pipeline_version.get_preproc_fn()).joinpath(process_type)

    def proc():
        PipelineSpecGenerator.SPEC_GENERATOR[process_type](
            in_dir=in_dir,
            out_dir=out_dir,
            pre_proc_config=config_dict['preproc_config'][pipeline_version.pre_processing_ver],
            transform_config=config_dict['transform_config'][pipeline_version.transform_ver],
            num_workers=num_workers
        ).generate()

    if not os.path.exists(out_dir):
        proc()


def audio_meta_extraction(pipeline_version:PipelineVersion, process_type: str, config_dict: dict, num_workers: int):
    out_dir = Path(config_dict['extractor_dir']).joinpath(
        pipeline_version.get_extractor_fn()).joinpath(process_type)

    Extractor.EXTRACTOR[process_type](
        x_in_dir=Path(config_dict['handcrafted_feats_dir']).joinpath(pipeline_version.get_transform_fn(), process_type),
        y_in_dir=Path(config_dict['db_audio_dir']).joinpath(pipeline_version.get_audio_db_fn(), process_type),
        out_dir=out_dir,
        pre_proc_config=config_dict['preproc_config'][pipeline_version.pre_processing_ver],
        transform_config=config_dict['transform_config'][pipeline_version.transform_ver],
        extractor_config=config_dict[f'{process_type}_extraction_config'][pipeline_version.extractor_ver],
        num_workers=num_workers
    ).extract()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()

    p.add_argument('-p', '--process', type=str, choices=['music_detection', 'audio_meta','audio_analysis'])
    p.add_argument('-m', '--mode', type=str, choices=['db_audio_gen', 'preprocessing', 'db_spectrum_gen', 'extraction', 'all'])
    p.add_argument('-g', '--gpus', type=str, default='all')
    p.add_argument('-c', '--config_path', type=str, default='./assets/config/config.yaml')
    p.add_argument('-v', '--version', type=str)
    p.add_argument('--num_workers', type=int, default=2)
    arg = p.parse_args()

    with open(arg.config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    if arg.gpus != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpus

    pipeline_version = PipelineVersion.from_str(arg.version)

    if arg.mode == 'db_audio_gen':
        db_audio_gen(pipeline_version=pipeline_version, process_type=arg.process, config_dict=config_dict, num_workers=arg.num_workers)
    elif arg.mode == 'preprocessing':
        db_preprocessing(pipeline_version=pipeline_version, process_type=arg.process, config_dict=config_dict, num_workers=arg.num_workers)
    elif arg.mode == 'db_spectrum_gen':
        db_handcrafted_feats_gen(pipeline_version=pipeline_version, process_type=arg.process, config_dict=config_dict, num_workers=arg.num_workers)
    elif arg.mode == 'extraction':
        audio_meta_extraction(pipeline_version=pipeline_version, process_type=arg.process, config_dict=config_dict, num_workers=arg.num_workers)
    else: # all
        db_audio_gen(pipeline_version=pipeline_version, process_type=arg.process, config_dict=config_dict, num_workers=arg.num_workers)
        db_preprocessing(pipeline_version=pipeline_version, process_type=arg.process, config_dict=config_dict, num_workers=arg.num_workers)
        db_handcrafted_feats_gen(pipeline_version=pipeline_version, process_type=arg.process, config_dict=config_dict, num_workers=arg.num_workers)
        audio_meta_extraction(pipeline_version=pipeline_version, process_type=arg.process, config_dict=config_dict, num_workers=arg.num_workers)

