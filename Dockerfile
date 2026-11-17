# Full pipeline image: alignment, PosMix simulation, somatic callers,
# SomaticXGB training / prediction, errorRates and figure scripts.
#
# Build:
#   docker build -t cross-plat-somatic-snv .
#
# Run:
#   docker run --rm -v /path/to/data:/data cross-plat-somatic-snv doctor
#   docker run --rm -v /path/to/data:/data cross-plat-somatic-snv align --cores 64
#

FROM condaforge/miniforge3:24.9.2-0

LABEL org.opencontainers.image.title="cross-plat-somatic-snv"
LABEL org.opencontainers.image.description="Reproduction of the cross-platform low-frequency somatic SNV benchmark"
LABEL org.opencontainers.image.source="https://github.com/liqg/cross-plat-somatic-snv"

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_NO_BANNER=1 \
    PATH=/opt/conda/bin:$PATH

# The base image is Ubuntu; APT_MIRROR replaces archive.ubuntu.com /
# security.ubuntu.com (e.g. mirrors.aliyun.com on restricted networks).
ARG APT_MIRROR=archive.ubuntu.com
RUN set -eux; \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do \
        [ -f "$f" ] || continue; \
        sed -i "s|http://archive.ubuntu.com|http://${APT_MIRROR}|g; \
                s|http://security.ubuntu.com|http://${APT_MIRROR}|g; \
                s|http://deb.debian.org|http://${APT_MIRROR}|g" "$f"; \
    done; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get -o Acquire::ForceIPv4=true update && apt-get -o Acquire::ForceIPv4=true install -y --no-install-recommends \
        ca-certificates curl git procps zlib1g-dev libbz2-dev liblzma-dev \
        libcurl4-openssl-dev libssl-dev fonts-liberation poppler-utils ghostscript proxychains dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Optional: send mamba/conda traffic through a local HTTP proxy.  Build with
#   --build-arg PROXY_HOST=<proxy-host> --build-arg PROXY_PORT=1081
# when the container cannot reach conda.anaconda.org directly.
ARG PROXY_HOST=
ARG PROXY_PORT=1081
RUN if [ -n "${PROXY_HOST}" ]; then \
        printf 'strict_chain\nproxy_dns\n[ProxyList]\nhttp %s %s\n' \
            "${PROXY_HOST}" "${PROXY_PORT}" > /etc/proxychains.conf; \
    fi

# Create a dedicated environment instead of downgrading the base environment
# from Python 3.12 to 3.10, which makes the solver explore the whole base stack.
COPY docker/environment.yml /tmp/environment.yml
COPY docker/environment-legacy.yml /tmp/environment-legacy.yml
RUN conda config --set channel_priority strict; \
    if [ -n "${PROXY_HOST}" ]; then \
        proxychains mamba env create -y -n cpsnv -f /tmp/environment.yml \
        && proxychains mamba env create -y -n legacy -f /tmp/environment-legacy.yml; \
    else \
        mamba env create -y -n cpsnv -f /tmp/environment.yml \
        && mamba env create -y -n legacy -f /tmp/environment-legacy.yml; \
    fi \
    && mamba clean -a -y
ENV PATH=/opt/conda/envs/cpsnv/bin:/opt/conda/envs/legacy/bin:/opt/conda/bin:$PATH

# Vendored R package used to read VCFs in the feature extractor
COPY software/R/data.vcf_0.2.0.tar.gz /opt/cpsnv/software/R/
RUN R CMD INSTALL /opt/cpsnv/software/R/data.vcf_0.2.0.tar.gz

WORKDIR /opt/cpsnv
COPY . /opt/cpsnv

# Our binaries (PosMix, posfeature, bam-readcount, errorRates) and the VarScan jar.
ENV PATH=/opt/cpsnv/software/bin:$PATH \
    LD_LIBRARY_PATH=/opt/cpsnv/software/bin
RUN chmod +x software/bin/* software/errorRates/errorRates-v1.3 software/errorRates/run_errorRates.sh 2>/dev/null || true \
    && ln -sf /opt/cpsnv/software/varscan/VarScan.v2.4.3.jar /opt/conda/envs/cpsnv/bin/VarScan.v2.4.3.jar \
    && ln -sf /opt/cpsnv/software/errorRates/errorRates-v1.3 /opt/conda/envs/cpsnv/bin/errorRates-v1.3

# Fail early if a core tool is missing.
RUN fastp --version && bwa 2>&1 | head -1 && samtools --version | head -1 \
    && bcftools --version | head -1 && gatk --version | head -1 \
    && Rscript -e 'library(xgboost); library(data.table); library(data.vcf)' \
    && python -c 'import pandas, numpy, sklearn, xgboost, statsmodels, matplotlib'

ENTRYPOINT ["/opt/conda/envs/cpsnv/bin/python", "/opt/cpsnv/workflow/run.py"]
CMD ["doctor"]
