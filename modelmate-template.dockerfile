# openEuler 24.03 LTS-SP3 版本。
# 基础镜像必须包含
# /usr/local/python3.12.13/bin/python3.12，否则复制过来的 venv 无法启动。
# CANN / Ascend 运行库由基础镜像提供，不要再把 /usr/local/Ascend 打进本镜像。
# 若基础镜像是私有镜像，请先按平台文档推送到 ModelMate 镜像仓库。
#
# 构建上下文是本文件所在目录 /home/naie/work/docker。
# 先把当前 venv 以及它在 venv 之外引用的源码树放进上下文，再构建：
#   bash /home/naie/work/docker/stage_image_context.sh
#   docker build -f template.dockerfile -t uni-agent-train:YYYYMMDD .
#
# 上一版只 COPY 了 venv。venv 里这些包是 editable 安装，import 时读的是下面这些绝对路径，
# 缺了它们镜像里的 python 无法使用：
#   /home/work/user-job-dir/app/lib/py/uni-agent
#   /home/work/user-job-dir/app/lib/py/uni-agent/verl
#   /home/work/user-job-dir/app/lib/py/third_party/Megatron-LM
#   /home/work/user-job-dir/app/lib/py/third_party/Megatron-Bridge
#   /home/work/user-job-dir/app/lib/py/third_party/MegatronAdaptor
#   /home/work/user-job-dir/app/lib/py/third_party/MindSpeed
#   /home/work/user-job-dir/app/lib/py/third_party/MindSpeed-Bridge
#   /home/work/user-job-dir/app/lib/py/third_party/MindSpeed-Ops
#   /home/work/user-job-dir/app/lib/py/third_party/TransformerEngineNPU
#   /home/work/user-job-dir/app/lib/py/vllm-ascend-0.23.0-a3
# 本机 /home/naie/work 是指向 /home/work/user-job-dir/app 的符号链接，镜像里也按同样关系摆放。
FROM registry.fusionstage.local:20202/naie_2002/cann:9.1.0

# 平台模板中的 source 和 [[ ... ]] 需要 bash。
SHELL ["/bin/bash", "-c"]

################################ ↓制作镜像时DockerFile里必须执行的指令内容↓ ################################
################################  指令内容不可改，位置可根据实际需要调整  ################################

# OPTIONAL_FEATURES：all / none / debug,S3Storage,interactive
ENV OPTIONAL_FEATURES=all
ENV INSTALLATION_MODE=online

# 使用当前 Notebook 中已创建的 Python 3.12 venv。
ENV USER_PYTHON3_HOME=/home/naie/work/lib/py/venv
ENV USER_PYTHON_PKG_PATH=/home/naie/work/lib/py/venv/lib/python3.12/site-packages
ENV VENV_BASE_PYTHON=/usr/local/python3.12.13/bin/python3.12

# 使用平台服务时，全部为 naie 用户。
ENV NB_USER=naie
ENV NB_UID=17000
ENV NB_GROUP=naie
ENV NB_GID=17000
ENV JAVA_HOME=/home/$NB_USER/jre
ENV CLASSPATH=${JAVA_HOME}/lib

# 平台随附的构建材料；请先执行：
# cp -R /home/naie/material/* /home/naie/work/docker/
COPY common/script_for_build_img /home/
USER root
RUN sh /home/pre_execution_by_root.sh

# pre_execution_by_root.sh 已创建 /home/work/user-job-dir/app 和 naie 用户。
# editable 安装写死的是 /home/work/user-job-dir/app，平台变量写的是 /home/naie/work。
RUN ln -sfn /home/work/user-job-dir/app /home/naie/work

# 构建前由 stage_image_context.sh 把现有环境复制到本目录。
# 目标目录必须是这次脚本新建的，不要在已有 venv/ 上再 cp -a，否则会得到 venv/venv。
COPY --chown=naie:naie venv/ /home/work/user-job-dir/app/lib/py/venv/
COPY --chown=naie:naie third_party/ /home/work/user-job-dir/app/lib/py/third_party/
COPY --chown=naie:naie uni-agent/ /home/work/user-job-dir/app/lib/py/uni-agent/
COPY --chown=naie:naie vllm-ascend-0.23.0-a3/ /home/work/user-job-dir/app/lib/py/vllm-ascend-0.23.0-a3/

# 尽早确认该 venv 与所选基础镜像兼容，并确认 editable 源码树在镜像里的绝对路径可用。
RUN test -x "$VENV_BASE_PYTHON" || { \
      echo "ERROR: base image is missing $VENV_BASE_PYTHON" >&2; \
      exit 1; \
    }; \
    test -x "$USER_PYTHON3_HOME/bin/python" && \
    test -d "$USER_PYTHON_PKG_PATH" && \
    test -L /home/naie/work && \
    "$USER_PYTHON3_HOME/bin/python" -c \
      'import os, sys; \
assert sys.version_info[:3] == (3, 12, 13), sys.version; \
req = [ \
  "/home/work/user-job-dir/app/lib/py/uni-agent/uni_agent", \
  "/home/work/user-job-dir/app/lib/py/uni-agent/verl/verl", \
  "/home/work/user-job-dir/app/lib/py/third_party/Megatron-LM/megatron/core", \
  "/home/work/user-job-dir/app/lib/py/third_party/Megatron-Bridge/src/megatron/bridge", \
  "/home/work/user-job-dir/app/lib/py/third_party/MegatronAdaptor/megatron_adaptor", \
  "/home/work/user-job-dir/app/lib/py/third_party/MindSpeed/mindspeed", \
  "/home/work/user-job-dir/app/lib/py/third_party/MindSpeed-Bridge/mindspeed_bridge", \
  "/home/work/user-job-dir/app/lib/py/third_party/MindSpeed-Ops/mindspeed_ops", \
  "/home/work/user-job-dir/app/lib/py/third_party/TransformerEngineNPU/transformer_engine", \
  "/home/work/user-job-dir/app/lib/py/vllm-ascend-0.23.0-a3/vllm_ascend", \
]; \
missing = [p for p in req if not os.path.isdir(p)]; \
assert not missing, missing; \
needles = ("lib/py/uni-agent", "lib/py/uni-agent/verl"); \
absent = [n for n in needles if not any(p.rstrip("/").endswith(n) for p in sys.path)]; \
assert not absent, (absent, sys.path); \
print(sys.executable, sys.version); \
print("editable source trees ok")'

RUN echo "====================The value of USER_PYTHON3_HOME is $USER_PYTHON3_HOME" && \
    if [ ! -e "$USER_PYTHON3_HOME/bin/python" ] && [ -e "$USER_PYTHON3_HOME/bin/python3" ]; then \
      ln -s "$USER_PYTHON3_HOME/bin/python3" "$USER_PYTHON3_HOME/bin/python"; \
    fi && \
    if [[ $USER_PYTHON3_HOME == "/home/naie"* ]]; then \
      PYTHON3_HOME=$USER_PYTHON3_HOME; \
      PYTHON_PKG_PATH=$USER_PYTHON_PKG_PATH; \
    elif [[ $USER_PYTHON3_HOME != "/root"* ]]; then \
      PYTHON3_HOME=$USER_PYTHON3_HOME; \
      PYTHON_PKG_PATH=$USER_PYTHON_PKG_PATH; \
      usermod -aG root $NB_USER; \
      chown $NB_USER:$NB_GROUP $PYTHON3_HOME/bin; \
      chown -R $NB_USER:$NB_GROUP $PYTHON_PKG_PATH; \
    else \
      user_envs_root_path=/home/$NB_USER/user_envs; \
      mkdir -p $user_envs_root_path; \
      cp -a $USER_PYTHON3_HOME $user_envs_root_path; \
      python_basename=$(basename $USER_PYTHON3_HOME); \
      PYTHON3_HOME=$user_envs_root_path/$python_basename; \
      PYTHON_PKG_PATH=$PYTHON3_HOME/lib/python3.*/site-packages; \
      find $PYTHON3_HOME/bin -type f -exec sed -i '1s@^#!.*python.*@#!'"$PYTHON3_HOME"'/bin/python@' {} +; \
      chown -R $NB_USER:$NB_GROUP $user_envs_root_path; \
    fi && \
    if [[ $USER_PYTHON3_HOME == "/usr" ]]; then \
      chown -R $NB_USER:$NB_GROUP $USER_PYTHON3_HOME/lib*/python3.*/site-packages; \
    fi && \
    echo "export PYTHON3_HOME=$PYTHON3_HOME" >> /home/"${NB_USER}"/.bashrc && \
    echo "export PYTHON_PKG_PATH=$PYTHON_PKG_PATH" >> /home/"${NB_USER}"/.bashrc && \
    echo "export PATH=$PYTHON3_HOME/bin:/home/${NB_USER}/.local/bin:${PATH}" >> /home/"${NB_USER}"/.bashrc

# 平台可选能力依赖。
COPY --chown=naie:naie common/utils /home/install/
COPY --chown=naie:naie common/pip.conf /home/$NB_USER/.pip/
USER $NB_USER
RUN source /home/$NB_USER/.bashrc && sh /home/install/pre_execution_by_naie.sh

################################ ↑制作镜像时DockerFile里必须执行的指令内容↑ ################################

# 业务能力 DIY 区域↓
USER root

# Build nodes may not resolve the public openEuler repositories. Place the matching
# openEuler 24.03 LTS-SP3 aarch64 RPM beside this Dockerfile as patch.rpm.
COPY patch.rpm /tmp/patch.rpm
RUN if ! command -v patch >/dev/null 2>&1; then \
      rpm -Uvh /tmp/patch.rpm; \
    fi && \
    command -v patch && patch --version | head -n 1 && \
    rm -f /tmp/patch.rpm

# 业务能力 DIY 区域↑

################################ ↓制作镜像时DockerFile里需最后执行的指令内容↓ ################################
WORKDIR /home/$NB_USER/
USER $NB_USER
################################ ↑制作镜像时DockerFile里需最后执行的指令内容↑ ################################
