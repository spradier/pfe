Loïc, pour la jetson :

**Commence par suivre toutes les instructions dans l’ordre de ce lien, tout est expliqué pour setup le jetson**
https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

**Ensuite tape les commandes classiques :**

```
$ sudo apt-get update
$ sudo apt install python
$ sudo apt install python3
$ sudo apt install python-pip
$ sudo apt install python3-pip
```

**Vérifie le port de ta caméra**

`$ v4l2-ctl --list-devices`


**Ensuite tu as juste à lancer le programme (pour l'instant, tu lances test.py le temps que je mette client.py à jour) en pensant à choisir la bonne caméra (pour l'instant faut que tu modifies le code source du client en remplaçant le 2 par le port que tu utilises (0 par défaut)) :**

Etienne ne t'occupe pas de ça pour l'instant, on n'a pas besoin d'openCV
** Setup OpenCV

```
$ sudo apt-get install libfreetype6-dev python3-setuptools
$ sudo apt-get install protobuf-compiler libprotobuf-dev openssl
$ sudo apt-get install libssl-dev libcurl4-openssl-dev
$ sudo apt-get install cython3

$ sudo apt-get install build-essential pkg-config
$ sudo apt-get install libtbb2 libtbb-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
$ sudo apt-get install libxvidcore-dev libavresample-dev
$ sudo apt-get install libtiff-dev libjpeg-dev libpng-dev
$ sudo apt-get install python-tk libgtk-3-dev
$ sudo apt-get install libcanberra-gtk-module libcanberra-gtk3-module
$ sudo apt-get install libv4l-dev libdc1394-22-dev

$ wget https://raw.githubusercontent.com/jkjung-avt/jetson_nano/master/install_protobuf-3.6.1.sh
$ sudo chmod +x install_protobuf-3.6.1.sh
$ ./install_protobuf-3.6.1.sh

$ cd ~
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.2.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.2.zip
$ unzip opencv.zip
$ unzip opencv_contrib.zip
$ mv opencv-4.1.2 opencv
$ mv opencv_contrib-4.1.2 opencv_contrib
$ cd opencv
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D WITH_CUDA=ON \
	-D CUDA_ARCH_PTX="" \
	-D CUDA_ARCH_BIN="5.3,6.2,7.2" \
	-D WITH_CUBLAS=ON \
	-D WITH_LIBV4L=ON \
	-D BUILD_opencv_python3=ON \
	-D BUILD_opencv_python2=OFF \
	-D BUILD_opencv_java=OFF \
	-D WITH_GSTREAMER=ON \
	-D WITH_GTK=ON \
	-D BUILD_TESTS=OFF \
	-D BUILD_PERF_TESTS=OFF \
	-D BUILD_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_EXTRA_MODULES_PATH=/home/`whoami`/opencv_contrib/modules ..
$ make -j4
$ sudo make install
$ cd ~/.virtualenvs/$NOM_ENVIRONNEMENT_VIRTUEL$/lib/python3.6/site-packages/   //Remplace $NOM_ENVIRONNEMENT_VIRTUEL$ par le nom de ton environnement virtuel
$ ln -s /usr/local/lib/python3.6/site-packages/cv2/python3.6/cv2.cpython-36m-aarch64-linux-gnu.so cv2.so
```



