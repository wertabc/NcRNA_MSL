# **NcRNA_MSL:A Novel Method of Multimodal Stable Learning for Non-Coding RNA Prediction in Human, Mouse, and Plant**
Long non-coding RNAs (lncRNAs) play pivotal regulatory roles in cellular processes, significantly influencing gene expression, genome stability, and cell differentiation. 
Accurate identification of lncRNAs is essential for elucidating their biological functions, necessitating the development of advanced prediction methods. 
Building on previous research in our laboratory, NcRNA_MSL introduces a stable learning strategy. It is trained, validated, and tested on human, mouse, and plant non-coding RNA datasets, demonstrating robust performance. Furthermore, the NcRNA_MSL framework offers a novel multimodal stable learning approach, paving the way for its application in other types of gene identification.

# **AUTHOR/SUPPORT**
* Mengqing Gao - GaoMengqing@ahau.stu.edu.cn
* Xiu Jin - jiaotong_jin@163.com
* Zhengyang Chen - czy20040511@gmail.com
* Corresponding author.Xiaodan Zhang - zxdahau@163.com


# **NcRNA_MSL**
## Requiredpackages ##
* python==3.9.2(or a compatible version)
* numpy==1.21.3(or a compatible version)
* pandas==1.3.4(or a compatible version)
* scikit-learn==1.1.3(or a compatible version)
* torch==1.13.1(or a compatible version)

## Online Usage Platform ##
For more intuitive use of tools, we provide a user-friendly web platform for non-coding RNA identification, available at [SPT-IFL HOME (sptifl-ncrnapre.com)](http://www.sptifl-ncrnapre.com/).
### 1.How to install SPT-IFL
#### 1.1Run SPT-IFL from docker(Locally、Simply)
##### 1.1.1 Make sure the machine contains a Docker environment (https://www.docker.com/get-started/).
##### 1.1.2 Download the SPT-IFL and add the data file to the project directory
* git https://github.com/wertabc/SPT-IFL
* cd SPT-IFL
* (upload the data file,example:data.fasta)
##### 1.1.3 Pull and build the environment image.(Time required)
* sudo docker build -t pinc_images .
##### 1.1.4 Create and enter a new container.
* sudo docker run -it pinc_images bash
##### 1.1.5 Execute SPT-IFL for prediction
* python pinc.py -f data.fasta

### 2. Run SPT-IFL from source code(Complex)
#### 2.1 Installation Environment(Autogluon、kentUtils).
#### 2.2 Clone project, install related dependencies.
#### 2.3 Execute SPT-IFL for prediction.
