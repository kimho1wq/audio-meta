#!/bin/bash

# echo with host name
function echo_host {
  echo "[$(hostname)] $1"
}

echo_host "Installing audio meta server"


sudo yum -y install java-11-openjdk-devel.x86_64
sudo yum -y install gcc openssl-devel zlib-devel libffi-devel bzip2-devel
sudo yum -y install wget
sudo yum -y install make
sudo yum -y install git
sudo yum -y install curl
sudo yum -y install epel-release
sudo yum -y localinstall --nogpgcheck https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm 
sudo yum -y install ffmpeg ffmpeg-devel 

wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz
tar xzf Python-3.8.10.tgz
rm Python-3.8.10.tgz
cd Python-3.8.10
./configure --enable-optimizations
sudo make altinstall

sudo ln -Tfs /usr/local/bin/python3.8 /usr/bin/python
sudo ln -s /usr/local/bin/pip3.8 /usr/bin/pip
sudo ln -Tfs /usr/lib/jvm/java-11-openjdk-11.0.12.0.7-0.el7_9.x86_64/bin/java /etc/alternatives/java
sudo ln -Tfs /usr/share/man/man1/java-java-11-openjdk-11.0.12.0.7-0.el7_9.x86_64.1.gz /etc/alternatives/java.1.gz
sudo ln -Tfs /usr/lib/jvm/java-11-openjdk-11.0.12.0.7-0.el7_9.x86_64/bin/javac /etc/alternatives/javac
sudo ln -Tfs /usr/share/man/man1/javac-java-11-openjdk-11.0.12.0.7-0.el7_9.x86_64.1.gz /etc/alternatives/javac.1.gz
sudo ln -Tfs /usr/lib/jvm/java-11-openjdk-11.0.12.0.7-0.el7_9.x86_64/bin/javadoc /etc/alternatives/javadoc
sudo ln -Tfs /usr/share/man/man1/javadoc-java-11-openjdk-11.0.12.0.7-0.el7_9.x86_64.1.gz /etc/alternatives/javadoc.1.gz
sudo ln -Tfs /usr/lib/jvm/java-11-openjdk-11.0.12.0.7-0.el7_9.x86_64/bin/javap /etc/alternatives/javap
sudo ln -Tfs /usr/share/man/man1/javap-java-11-openjdk-11.0.12.0.7-0.el7_9.x86_64.1.gz /etc/alternatives/javap.1.gz
sudo ln -Tfs /usr/lib/jvm/java-11-openjdk-11.0.12.0.7-0.el7_9.x86_64 /etc/alternatives/java_sdk

pip install --upgrade pip
pip install numpy==1.20.0
pip install torch==1.9.0
pip install torchvision==0.10.0
pip install torchtext==0.10.0
pip install torchaudio==0.9.0
pip install librosa==0.8.1
pip install PyYAML==5.3.1
pip install pydub==0.25.1
pip install future
pip install psutil
pip install wheel
pip install requests
pip install sentencepiece
pip install pillow==8.2.0
pip install captum
pip install packaging
pip install setuptools 
pip install pytorch_lightning
pip install torchserve


