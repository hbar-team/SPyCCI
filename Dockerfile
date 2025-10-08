# syntax=docker/dockerfile:1.7
FROM condaforge/mambaforge:24.9.2-0
SHELL ["/bin/bash","-o","pipefail","-c"]

# --- base system + tooling ---------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates tar zip xz-utils curl jq && \
    rm -rf /var/lib/apt/lists/*

# --- conda env ---------------------------------------------------------------
COPY environment.yml /tmp/
RUN mamba env create -f /tmp/environment.yml && mamba clean -afy
ENV CONDA_DEFAULT_ENV=spycci
ENV PATH=/opt/conda/envs/spycci/bin:$PATH

# --- project sources ---------------------------------------------------------
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -e .

# --- DFTB+ parameters --------------------------------------------------------
ENV DFTBPLUS_PARAM_DIR=/opt/dftbplus/slakos
RUN mkdir -p "${DFTBPLUS_PARAM_DIR}" && \
    wget -q https://github.com/dftbparams/3ob/releases/download/v3.1.0/3ob-3-1.tar.xz -O /tmp/3ob.tar.xz && \
    tar -xf /tmp/3ob.tar.xz -C "${DFTBPLUS_PARAM_DIR}" && rm /tmp/3ob.tar.xz

# --- ORCA (private release asset) -------------------------------------------
ARG ORCA_OWNER=hbar-team
ARG ORCA_REPO=orca-binaries
ARG ORCA_TAG=v6.1.0-f.0
ARG ORCA_ASSET=orca-6.1.0-f.0_linux_x86-64_openmpi41.tar.xz
ARG ORCA_LOCAL_ARCHIVE=
RUN --mount=type=secret,id=gh_token,required=0 \
    set -euo pipefail; \
    TOKEN_FILE="/run/secrets/gh_token"; \
    ORCA_TMP="/tmp/orca.tar.xz"; \
    if [ -s "$TOKEN_FILE" ]; then \
    GH_TOKEN="$(cat "$TOKEN_FILE")"; \
    echo "Downloading ORCA via GitHub API (token detected)"; \
    release_json=$(curl --fail -s -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${ORCA_OWNER}/${ORCA_REPO}/releases/tags/${ORCA_TAG}"); \
    asset_api=$(echo "$release_json" | jq -r --arg NAME "$ORCA_ASSET" '.assets[]? | select(.name==$NAME) | .url'); \
    [ -n "$asset_api" ] && [ "$asset_api" != "null" ] || { echo "Asset $ORCA_ASSET not found"; exit 1; }; \
    curl --fail -sL -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/octet-stream" "$asset_api" -o "$ORCA_TMP"; \
    else \
    [ -n "${ORCA_LOCAL_ARCHIVE}" ] || { \
    echo "Provide ORCA_PAT (for CI) or --build-arg ORCA_LOCAL_ARCHIVE=<path-in-context>"; exit 1; }; \
    LOCAL_SRC="/workspace/${ORCA_LOCAL_ARCHIVE}"; \
    [ -f "$LOCAL_SRC" ] || { echo "Local ORCA archive not found at $LOCAL_SRC"; exit 1; }; \
    echo "Using local ORCA archive $LOCAL_SRC"; \
    cp "$LOCAL_SRC" "$ORCA_TMP"; \
    rm -f "$LOCAL_SRC"; \
    fi; \
    mkdir -p /opt/orca && \
    tar -xf "$ORCA_TMP" -C /opt/orca --strip-components=1 2>/dev/null; \
    rm "$ORCA_TMP"; \
    chmod -R a+rx /opt/orca; \
    find /opt/orca -maxdepth 2 -type f -name orca -perm -u+x -print -quit || { echo "ORCA binary not found in /opt/orca"; exit 1; }

# --- XTB (official binary) ---------------------------------------------------
ENV XTBHOME=/opt/xtb
RUN wget -q https://github.com/grimme-lab/xtb/releases/download/v6.7.0/xtb-6.7.0-linux-x86_64.tar.xz && \
    tar -xf xtb-6.7.0-linux-x86_64.tar.xz && \
    mv xtb-dist "${XTBHOME}" && \
    rm xtb-6.7.0-linux-x86_64.tar.xz

# --- CREST (official binary) -------------------------------------------------
ENV CREST_BIN=/opt/crest
RUN wget -q https://github.com/crest-lab/crest/releases/download/v3.0.2/crest-gnu-12-ubuntu-latest.tar.xz && \
    tar -xf crest-gnu-12-ubuntu-latest.tar.xz && \
    mv crest "${CREST_BIN}" && \
    rm crest-gnu-12-ubuntu-latest.tar.xz && \
    chmod +x ${CREST_BIN}/crest

# --- runtime env + user ------------------------------------------------------
ENV ORCA_ROOT=/opt/orca
ENV PATH=/opt/orca/bin:${XTBHOME}/bin:${CREST_BIN}:/opt/conda/envs/spycci/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/spycci/lib:/opt/orca/lib:$LD_LIBRARY_PATH
ENV OMPI_MCA_plm=isolated
ENV OMPI_MCA_rmaps_base_oversubscribe=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV OMPI_MCA_btl=^openib
ENV PYTHONUNBUFFERED=1

ENV SPYCCI_VERSION_MATCH=minor

RUN useradd -m -s /bin/bash spyccitest && \
    chown -R spyccitest:spyccitest \
    /workspace \
    /opt/conda/envs/spycci \
    "${ORCA_ROOT}" "${XTBHOME}" "${CREST_BIN}" "${DFTBPLUS_PARAM_DIR}" 
USER spyccitest
WORKDIR /workspace

CMD ["pytest","-vvv","--color=yes","-ra","--maxfail=0"]