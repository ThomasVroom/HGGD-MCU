pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install fvcore wheel & :: needed for compiling pytorch3d

:: pytorch3d needs to be compiled locally, can take 10-15 minutes
git clone https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation ./pytorch3d & :: no-build-isolation to use local torch

:: install other dependencies
pip install -r requirements.txt
