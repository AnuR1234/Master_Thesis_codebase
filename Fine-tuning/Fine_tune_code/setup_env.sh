#!/bin/bash

# Define colors for better visibility
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define the virtual environment name
VENV_NAME="llama3_env"

echo -e "${BLUE}========== Setting up environment for Llama 3 fine-tuning with DeepSpeed ==========${NC}"

# Install virtualenv if not already installed
if ! command -v virtualenv &> /dev/null; then
    echo -e "${YELLOW}Installing virtualenv...${NC}"
    pip install virtualenv
fi

# Create and activate virtual environment
echo -e "${YELLOW}Creating virtual environment '${VENV_NAME}'...${NC}"
virtualenv ${VENV_NAME}
echo -e "${GREEN}Virtual environment created successfully!${NC}"

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source ${VENV_NAME}/bin/activate
echo -e "${GREEN}Virtual environment activated!${NC}"

# Upgrade basic packages
echo -e "${YELLOW}Upgrading pip, setuptools, and wheel...${NC}"
pip install -U pip setuptools wheel
echo -e "${GREEN}Basic packages upgraded!${NC}"

# Install PyTorch with CUDA support
echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
pip install torch torchvision torchaudio

# Install main dependencies
echo -e "${YELLOW}Installing main dependencies...${NC}"
pip install transformers==4.36.2 datasets peft==0.7.1 trl==0.7.4 accelerate
echo -e "${GREEN}Main dependencies installed!${NC}"

# Install DeepSpeed
echo -e "${YELLOW}Installing DeepSpeed...${NC}"
pip install deepspeed==0.11.2
# Uncomment the line below for optimized installation if using CUDA 11.8
# DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=0 pip install deepspeed==0.11.2
echo -e "${GREEN}DeepSpeed installed!${NC}"

# Install data handling packages
echo -e "${YELLOW}Installing data handling packages...${NC}"
pip install numpy pandas pyarrow
echo -e "${GREEN}Data handling packages installed!${NC}"

# Install quantization libraries
echo -e "${YELLOW}Installing quantization libraries...${NC}"
pip install bitsandbytes>=0.41.0
echo -e "${GREEN}Quantization libraries installed!${NC}"

# Install visualization and utilities
echo -e "${YELLOW}Installing visualization and utilities...${NC}"
pip install matplotlib tensorboard evaluate scipy scikit-learn
echo -e "${GREEN}Visualization and utilities installed!${NC}"

# Optional: Install Flash Attention 2 (commented out by default)
# echo -e "${YELLOW}Installing Flash Attention 2...${NC}"
# pip install flash-attn --no-build-isolation
# echo -e "${GREEN}Flash Attention 2 installed!${NC}"

# Verify installations
echo -e "${YELLOW}Verifying installations...${NC}"
python -c "import torch; import transformers; import peft; import trl; import deepspeed; print('PyTorch version:', torch.__version__); print('Transformers version:', transformers.__version__); print('PEFT version:', peft.__version__); print('TRL version:', trl.__version__); print('DeepSpeed version:', deepspeed.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('Number of GPUs:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# Create a script to easily activate the environment later
echo -e "${YELLOW}Creating activation script...${NC}"
cat > activate_env.sh << EOL
#!/bin/bash
source ${VENV_NAME}/bin/activate
echo "Llama 3 fine-tuning environment activated!"
EOL
chmod +x activate_env.sh
echo -e "${GREEN}Activation script created! In the future, run './activate_env.sh' to activate the environment.${NC}"

echo -e "${GREEN}===========================================================================${NC}"
echo -e "${GREEN}Setup complete! Your virtual environment '${VENV_NAME}' is ready for Llama 3 fine-tuning.${NC}"
echo -e "${GREEN}The environment is currently activated. When you're done, type 'deactivate' to exit.${NC}"
echo -e "${GREEN}===========================================================================${NC}"

# Instructions for running
echo -e "${BLUE}To start training, run:${NC}"
echo -e "${YELLOW}./run_training.sh${NC}"