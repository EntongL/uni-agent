# openEuler 24.03 LTS-SP3 版本。
# 基础镜像必须包含
# /usr/local/python3.12.13/bin/python3.12，否则复制过来的 venv 无法启动。
# 若基础镜像是私有镜像，请先按平台文档推送到 ModelMate 镜像仓库。
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

# 构建前需把现有 venv 复制到构建上下文：
# cp -a /home/naie/work/lib/py/venv /home/naie/work/docker/venv
# pre_execution_by_root.sh 已创建 naie 用户，此时可按用户名设置属主。
COPY --chown=naie:naie venv/ /home/naie/work/lib/py/venv/

# 尽早确认该 venv 与所选基础镜像兼容，避免到 Notebook 启动阶段才暴露问题。
RUN test -x "$VENV_BASE_PYTHON" || { \
      echo "ERROR: base image is missing $VENV_BASE_PYTHON" >&2; \
      exit 1; \
    }; \
    test -x "$USER_PYTHON3_HOME/bin/python" && \
    test -d "$USER_PYTHON_PKG_PATH" && \
    "$USER_PYTHON3_HOME/bin/python" -c \
      'import sys; assert sys.version_info[:3] == (3, 12, 13), sys.version; print(sys.executable, sys.version)'

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

# 确保复制进镜像的 venv 可由平台用户读取和继续 pip install。
RUN chown -R "$NB_USER:$NB_GROUP" "$USER_PYTHON3_HOME"

# 业务能力 DIY 区域↑

################################ ↓制作镜像时DockerFile里需最后执行的指令内容↓ ################################
WORKDIR /home/$NB_USER/
USER $NB_USER
################################ ↑制作镜像时DockerFile里需最后执行的指令内容↑ ################################
