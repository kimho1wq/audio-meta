# custom handler file

"""
ModelHandler defines a custom model handler.
"""

from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import list_classes_from_module, load_label_mapping, map_class_to_label

import os
import yaml
import torch
import logging
import importlib.util

from assets.config.config import *
from src.preprocessing import Loader
from src.transform import AudioTransformer
from src.extraction.analysis import Analysis
from src.extraction.utils import get_input_shape, fill_input_arr, _c_to_r

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        super().__init__()

        with open('assets/config/config.yaml', 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        version = 'db1.pp0.tr0.ex1'
        pipeline_version = PipelineVersion.from_str(version)
        pre_proc_config=config_dict['preproc_config'][pipeline_version.pre_processing_ver]
        transform_config=config_dict['transform_config'][pipeline_version.transform_ver]
        extractor_config=config_dict['audio_meta_extraction_config'][pipeline_version.extractor_ver]
        analysis_config = config_dict['audio_analysis_extraction_config'][pipeline_version.extractor_ver]
        self.pre_proc_config = PreProcConfig.from_dict(pre_proc_config)
        self.transform_config = TransformConfig.from_dict(transform_config)
        self.extractor_config = ExtractorConfig.from_dict(extractor_config)
        self.analysis_config = AnalysisExtractorConfig.from_dict(analysis_config)

        self.audio_loader = Loader(self.pre_proc_config.audio_info)
        self.transform = [AudioTransformer.TRANSFORMER[t_config.transform_type](**t_config.param)
                      for t_config in self.transform_config.transform_info]
        self.input_shape = get_input_shape(self.pre_proc_config, self.transform_config, self.extractor_config)
        self.analyzer = Analysis()

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
            self.model.to(self.device)
        else:
            logger.debug("Loading torchscript model")
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")

            self.model = self._load_torchscript_model(model_pt_path)

        self.model.eval()

        logger.debug('Model file %s loaded successfully', model_pt_path)

        # Load class mapping for classifiers
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        self.mapping = load_label_mapping(mapping_file_path)

        self.initialized = True


    def _load_torchscript_model(self, model_pt_path):
        return torch.jit.load(model_pt_path, map_location=self.device)


    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError(
                "Expected only one class as model definition. {}".format(
                    model_class_definitions
                )
            )

        model_class = model_class_definitions[0]
        # Load model class
        #model = model_class()
        model = model_class(self.extractor_config.cnn_config,
            self.extractor_config.rnn_config,
            len(self.extractor_config.audio_features),
            self.input_shape
        )

        if model_pt_path:
            state_dict = torch.load(model_pt_path, map_location=self.device)
            model.load_state_dict(state_dict)
        
        return model

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        # Load audio file..

        y = self.audio_loader(data[0].get("body"))
        input_x = {}
        for idx, transformer in enumerate(self.transform):
            input_x[transformer.alias()] = transformer(
                y=y, sr=self.pre_proc_config.audio_info.sampling_rate
            ).astype('float32')

        return input_x

    def inference(self, x):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        #signal processing analyzer results
        transform_param = {}
        transform_info = self.transform_config.to_dict()['transform_info']
        for idx in transform_info:
            transform_param[transform_info[idx]['transform_type']] = transform_info[idx]['param']

        res = {'features':{},'anal':{}}
        for _anal in self.analysis_config.analysis_info:
            _anal.param.update(transform_param[_anal.target_transform])
            _anal.param.update(self.pre_proc_config.audio_info.to_dict())
            
            pred = self.analyzer(_anal.audio_feature, x[_anal.target_transform], _anal.param)

            if _anal.audio_feature=='pitch':
                res['anal'].update(pred)
            else:
                res['features'].update(pred)


        # Do some inference call to engine here and return output
        seq_len = self.input_shape[0]
        for key, val in x.items():
            x[key] = torch.from_numpy(fill_input_arr(seq_len, val)).to(self.device)

        # Do model inference
        y = self.model(x)
        preds = y.detach().cpu().tolist()[0]

        return preds, res

    def postprocess(self, preds, res):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        if self.extractor_config.task_config.task_type[0] == 'r':
            res['features'].update({
                self.extractor_config.audio_features[i]: float(preds[i][0]) \
                for i in range(len(preds))
            })
        else:
            res['features'].update({
                self.extractor_config.audio_features[i]: float(_c_to_r(preds[i], self.extractor_config.task_config.q_index)) \
                for i in range(len(preds))
            })
            res.update({
                self.extractor_config.audio_features[i]: [float(m) for m in preds[i]] \
                for i in range(len(preds))
            })

        return [res]


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        _input = self.preprocess(data)
        _output, res = self.inference(_input)
        return self.postprocess(_output, res)
