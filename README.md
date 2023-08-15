# Dress Me Up

We provide the demo code for **Dress Me Up: A Dataset & Method for Self-Supervised 3D Garment Retargeting**, 3DV 2023 submission. Check out the code at     [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11HgaGoTX7-k6uvmffuxUtzXVgb6aEftV?usp=sharing)

![](./images/teaser_new.png) 



## Installation
Install required libraries using following command
```
pip install -r requirements.txt
```

Install [Mesh](https://github.com/MPI-IS/mesh) package
```
apt-get install libboost-dev
git clone https://github.com/MPI-IS/mesh.git
%cd mesh
BOOST_INCLUDE_DIRS=/path/to/boost/include make all
%cd ..
```

## Demo
The input garment and meshes are present in "dataset" folder. Run the following code to drape garment on to given scan. The results are saved in "./results".
```
python trainer.py 
```


You can also tryout the Jupyter or Colab Notebook provided to run the code and visualization.
