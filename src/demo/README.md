# Instructions for AR

![FrameNet AR Teaser](https://github.com/hjwdzh/framenet/raw/master/img/teaser-ar.jpg)

### Build the cpp library
```
cd cpp
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Download the data
```
sh download.sh
```

### Run the App
Our app takes the scene and the object as input. Please see our default options in the script for example.
```
python AttachTexture.py [--input scene] [--resource object]
```

### Instructions for usage
* Move mouse to specify the center the object is going to be placed.
* Left click to place the object.
* Press 'g' to quit the app.
* Press 'd' and 'f' to make the object larger or smaller.
* Press 'a' to switch among three different modes.
  * Attach pattern in rigid mode.
  * Attach pattern in deformable mode.
  * Attach a 3D object.
* Press 'r' to rotate the object among 4 possible orientations.
