{
  pkgs,
  lib,
  stdenv,
  ...
}:
let
  pkgs-unfree = import pkgs.path {
    inherit (pkgs) system;
    config.allowUnfree = true;
  };
  pythonPackages = pkgs-unfree.python312Packages;
in
pkgs-unfree.mkShell {
  buildInputs = [
    pythonPackages.python
    pythonPackages.venvShellHook
    pkgs-unfree.autoPatchelfHook
    pythonPackages.datasets
    pythonPackages.evaluate
    pythonPackages.jiwer
    pythonPackages.accelerate
    pythonPackages.tensorboard
    pythonPackages.scikit-learn
    pythonPackages.pillow
    pythonPackages.transformers
    pkgs-unfree.cudaPackages.cudatoolkit
    pkgs-unfree.cudaPackages.cudnn
    pkgs-unfree.linuxPackages.nvidia_x11
  ];
  venvDir = "./.venv";
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH    
    autoPatchelf ./.venv   '';
  postShellHook = ''
    unset SOURCE_DATE_EPOCH     
    export LD_LIBRARY_PATH=${pkgs-unfree.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH     
    export LD_LIBRARY_PATH=${pkgs-unfree.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs-unfree.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${lib.makeLibraryPath [ stdenv.cc.cc ]}:$LD_LIBRARY_PATH '';
}
