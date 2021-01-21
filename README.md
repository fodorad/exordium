# utils
Collection of utility functions and deep learning methods.

--------
## Install dependencies
```bash
pip install -r requirements.txt --user
```

Add openSMILE binary to PATH and the parent directory of opensmile_wrapper to PYTHONPATH system variables.
```bash
git submodule update --init --recursive
tar -xvzf ./tools/opensmile_wrapper/opensmile-2.3.0.tar.gz -C ./tools/opensmile_wrapper
export PATH="$PWD/tools/opensmile_wrapper/opensmile-2.3.0/bin/linux_x64_standalone_static/:$PATH"
export PYTHONPATH="$PWD/tools/:$PYTHONPATH"
```

Install pyAudioAnalysis package
```bash
mv ./tools/pyAudioAnalysis/requirements.txt ./tools/pyAudioAnalysis/_requirements.txt 
pip install -e ./tools/pyAudioAnalysis
```

Docker image for pose estimation
paper: https://arxiv.org/abs/1611.08050
```bash
docker run -it -p 5000:5000 quay.io/codait/max-human-pose-estimator
```

Katna for keyframe detection
```bash
cd tools/Katna && bash install.sh && cd ../..
```