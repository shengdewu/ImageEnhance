## trilinear

### build cpp trilinear

By default, we use pytorch 1.x:

    cd trilinear/cpp
    sh setup.torch1.x.sh
    
For pytorch 0.4.1:

    cd trilinear/cpp
    sh make.sh

### Use cpp trilinear 
    from .cpp.TrilinearInterpolationFunction import TrilinearInterpolationFunction
      
### Use python trilinear 
    from .python.TrilinearInterpolationFunction import TrilinearInterpolationFunction