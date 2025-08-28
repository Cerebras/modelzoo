#!/bin/bash

# ---------- Argument Parsing ----------
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <path_to_virtual_environment_directory> <path_to_requirements_file>"
    echo "Error: Missing arguments."
    exit 1
fi
VENV_DIR="$1" 
REQUIREMENTS_FILE="$2"

if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    echo "Error: Requirements file not found at ${REQUIREMENTS_FILE}"
    exit 1
fi

# ---------- Virtual Environment Setup ----------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if venv already exists. If not, create it.
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment in ${VENV_DIR}..."
    python -m venv "${VENV_DIR}"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
else
    echo "Virtual environment ${VENV_DIR} already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

# ---------- Toolchain SDK Setup ----------

# Define the base directory for the toolchain SDK.
# This can be overridden by setting TOOLCHAIN_SDK_BASE_DIR in your environment
# before sourcing this script.
: "${TOOLCHAIN_SDK_BASE_DIR:=${GITTOP}/toolchain/sdk-x86_64}"

if [ ! -d "${TOOLCHAIN_SDK_BASE_DIR}" ]; then
    echo "Error: TOOLCHAIN_SDK_BASE_DIR not found: ${TOOLCHAIN_SDK_BASE_DIR}"
    exit 1
fi

# Derive specific paths from the TOOLCHAIN_SDK_BASE_DIR
export SYSROOT_PATH="${TOOLCHAIN_SDK_BASE_DIR}/x86_64-buildroot-linux-gnu/sysroot"

# Compilers
export CC="${TOOLCHAIN_SDK_BASE_DIR}/bin/x86_64-buildroot-linux-gnu-gcc"
export CXX="${TOOLCHAIN_SDK_BASE_DIR}/bin/x86_64-buildroot-linux-gnu-g++"

# Flags
export CFLAGS="--sysroot=${SYSROOT_PATH} -I${SYSROOT_PATH}/usr/include"
export CXXFLAGS="--sysroot=${SYSROOT_PATH} -I${SYSROOT_PATH}/usr/include"
export LDFLAGS="--sysroot=${SYSROOT_PATH} -L${SYSROOT_PATH}/usr/lib" 
export CPPFLAGS="--sysroot=${SYSROOT_PATH}"
export LIBRARY_PATH="${SYSROOT_PATH}/usr/lib" 

# Shared linker commands
export LDSHARED="${CC} -shared"
export LDCXXSHARED="${CXX} -shared"

# Python sysconfig data
PYTHON_VERSION_SUBDIR="python3.11"

export SYSCONFIG_DIR="${TOOLCHAIN_SDK_BASE_DIR}/lib/${PYTHON_VERSION_SUBDIR}"
export _PYTHON_SYSCONFIGDATA_NAME="_sysconfigdata__linux_x86_64-linux-gnu"
export PYTHONPATH="${SYSCONFIG_DIR}:${PYTHONPATH}"

echo "Toolchain environment configured with TOOLCHAIN_SDK_BASE_DIR: ${TOOLCHAIN_SDK_BASE_DIR}"

# ---------- Install Requirements ----------

echo "Installing requirements from ${REQUIREMENTS_FILE}..."

# We need to install wheel because we are using pip without build isolation
pip install wheel
pip install --no-build-isolation -r ${REQUIREMENTS_FILE}