import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='exordium',
    version='1.0.1',
    author='fodorad',
    author_email="foauaai@inf.elte.hu",
    license='MIT',
    packages=setuptools.find_packages(),
    url="https://github.com/fodorad/exordium",
    description='Collection of utility tools and deep learning methods.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11.5",
    install_requires=[
        'opencv-python',
        'Pillow',
        'tqdm',
        'matplotlib',
        'scipy',
        'decord',
        'torch',
        'transformers',
        'torchvision',
        'networkx', # exordium.visualization.graph
        'opensmile', # exordium.audio.smile
        'librosa', # exordium.audio.spectrogram
        'torchaudio', # exordium.audio.io
        'moviepy', # exordium.video.io
        'pandas', # exordium.video.openface
        'einops', # exordium.video.r2plus1d
        'onnx', # exordium.video.tddfa_v2
        'onnxruntime', # exordium.video.tddfa_v2
        'types-PyYAML', # exordium.video.tddfa_v2
        'deepface', # exordium.video.track
        'tensorflow', # deepface
        'batch-face @ git+https://github.com/elliottzheng/batch-face.git@master', # exordium.video.facedetector
    ],
)