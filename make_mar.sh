mkdir -p extra_files/assets/ extra_files/src/extraction/
cp -r assets/config/ extra_files/assets/
cp -r src/preprocessing/ extra_files/src/
cp -r src/transform/ extra_files/src/
cp -r src/extraction/utils.py extra_files/src/extraction/
cp -r src/extraction/analysis.py extra_files/src/extraction/
torch-model-archiver --model-name audio_meta --version 0.1 --serialized-file assets/network/regression/db1.pp0.tr0.ex1/final_epoch.pth --export-path ./ --extra-file extra_files --handler infer_handler.py
rm -rf extra_files
