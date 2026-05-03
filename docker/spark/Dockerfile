FROM eclipse-temurin:17-jdk

# Install required packages
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    procps \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3 and pip
RUN apt-get update && apt-get install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV SPARK_VERSION=3.5.8
ENV HADOOP_VERSION=3
ENV ICEBERG_VERSION=1.9.2
ENV SPARK_HOME=/opt/spark
ENV HOME=/home/spark
ENV JUPYTER_CONFIG_DIR=/home/spark/.jupyter
ENV JUPYTER_DATA_DIR=/home/spark/.local/share/jupyter
ENV JUPYTER_RUNTIME_DIR=/home/spark/.local/share/jupyter/runtime
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV SPARK_JARS_PACKAGES="org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.9.2"


# Install Python packages for Iceberg and notebooks
# QUAN TRỌNG: pyspark phải cùng version với Spark binary (SPARK_VERSION=3.5.8)
# Không dùng pyspark không có version pin vì pip sẽ cài bản mới nhất (4.x)
# gây lỗi "'JavaPackage' object is not callable"
RUN pip3 install --break-system-packages \
    pyiceberg[s3fs,pyarrow] \
    protobuf==5.26.1 \
    jupyterlab \
    notebook \
    ipykernel \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scipy \
    scikit-learn \
    "pyspark==${SPARK_VERSION}"

# Download and install Spark
RUN wget "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    -O "/tmp/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" && \
    tar -xzf "/tmp/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    && mv "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" $SPARK_HOME \
    && rm "/tmp/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"

# Download Iceberg jars
RUN wget "https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/${ICEBERG_VERSION}/iceberg-spark-runtime-3.5_2.12-${ICEBERG_VERSION}.jar" \
    -O $SPARK_HOME/jars/iceberg-spark-runtime-3.5_2.12-${ICEBERG_VERSION}.jar

# Download Iceberg AWS integration
# RUN wget "https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws/${ICEBERG_VERSION}/iceberg-aws-${ICEBERG_VERSION}.jar" \
#     -O $SPARK_HOME/jars/iceberg-aws-${ICEBERG_VERSION}.jar

# Download AWS SDK for S3 support
# V1 for HadoopFileIO
RUN wget "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.565/aws-java-sdk-bundle-1.12.565.jar" \
    -O $SPARK_HOME/jars/aws-java-sdk-bundle-1.12.565.jar

# V2 for S3FileIO
# RUN wget "https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.20.143/bundle-2.20.143.jar" \
#     -O $SPARK_HOME/jars/aws-sdk-v2-bundle-2.20.143.jar

# Download Iceberg AWS bundle
RUN wget "https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/1.9.2/iceberg-aws-bundle-1.9.2.jar" \
    -O $SPARK_HOME/jars/iceberg-aws-bundle-1.9.2.jar

# Download Hadoop AWS
RUN wget "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar" \
    -O $SPARK_HOME/jars/hadoop-aws-3.3.4.jar

# Create spark user
RUN useradd -m -d /home/spark -r -u 185 -g 0 spark && \
    mkdir -p $SPARK_HOME/work-dir && \
    mkdir -p /home/spark/.jupyter && \
    mkdir -p /home/spark/.local/share/jupyter/runtime && \
    chown -R spark:0 $SPARK_HOME && \
    chown -R spark:0 /home/spark && \
    chmod -R g+w $SPARK_HOME

# Copy entrypoint script
COPY entrypoint.sh /opt/entrypoint.sh

# Fix quyền + line ending
RUN apt-get update && apt-get install -y dos2unix && \
    dos2unix /opt/entrypoint.sh && \
    chmod +x /opt/entrypoint.sh && \
    chown spark:0 /opt/entrypoint.sh

WORKDIR $SPARK_HOME

RUN mkdir -p $SPARK_HOME/logs && \
    chown -R spark:0 $SPARK_HOME/logs && \
    chmod -R 775 $SPARK_HOME/logs
    
USER spark

EXPOSE 7077 8080 8081

ENTRYPOINT ["/opt/entrypoint.sh"]