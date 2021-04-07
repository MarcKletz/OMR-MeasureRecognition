let
  # Pin nixpkgs
  pkgs = import (builtins.fetchTarball {
    name = "nixpkgs-unstable-2021-04-03";
    url = "https://github.com/nixos/nixpkgs/archive/dbc780569131988fc5e01191425ab2a30882f4dd.tar.gz";
    sha256 = "1clmc9gyfw9fwxa9lpjkqhgq86z3p4qxj7krhkcasrkci54j6m6y";
  }) {
    config.allowUnfree = true;
  };

  yacs = (pkgs.python3Packages.buildPythonPackage rec {
    pname = "yacs";
    version = "0.1.8";

    src = pkgs.python38Packages.fetchPypi {
      inherit pname version;
      sha256 = "1123j6ifgwb9sgadh4mcy00pvlmwk2pqkvh4m6z06c9bjhrcgi7g";
    };

    # I don't care
    doCheck = false;

    nativeBuildInputs = with pkgs.python38Packages; [
      pyyaml
    ];
  });

  iopath = (pkgs.python3Packages.buildPythonPackage rec {
    pname = "iopath";
    version = "0.1.8";

    # Thanks for being on Pypi, but without any uploads there
    # Thanks for not tagging versions on GitHub
    # Thanks for nothing.
    src = builtins.fetchGit {
      url = "https://github.com/facebookresearch/iopath.git";
      ref = "master";
      rev = "85bec7238f83f4381a18ad45d5eb804d3336fa64";
    };

    # A few tests do HTTP. One could disable them individually
    doCheck = false;

    propagatedBuildInputs = with pkgs.python38Packages; [
      tqdm
      portalocker
      pytorch
    ];
  });

  fvcore = (pkgs.python38Packages.buildPythonPackage rec {
    pname = "fvcore";
    version = "0.1.5.post20210402";

    src = pkgs.python3Packages.fetchPypi {
      inherit pname version;
      sha256 = "0zrb1dhiqdmjs6s36a2wrk4m4hry34zdy27l7rflwzc2q093rmji";
    };

    # There's an actual, proper test failure in here. Might blow up later
    doCheck = false;

    propagatedBuildInputs = with pkgs.python38Packages; [
      numpy
      yacs
      pyyaml
      tqdm
      termcolor
      pillow
      tabulate
      iopath
    ];
  });

  # Upstream: https://github.com/NixOS/nixpkgs/pull/90127
  pycocotools = pkgs.python38Packages.buildPythonPackage rec {
    pname = "pycocotools";
    version = "2.0.2";

    src = pkgs.python38Packages.fetchPypi {
      inherit pname version;
      sha256 = "06hz0iz4kqxhqby4j7bah8l41kg68bb118jawp172i4vg497lw94";
    };

    nativeBuildInputs = with pkgs.python38Packages; [
      cython
    ];
    
    buildInputs = with pkgs.python38Packages; [
      numpy
      matplotlib
    ];

    checkInputs = with pkgs.python38Packages; [
      numpy
      pytest
      matplotlib
    ];

    # No tests available on pypi.org
    doCheck = false;

    pythonImportsCheck = [ "pycocotools" ];

    meta = with pkgs.lib; {
      description = "COCO API - http://cocodataset.org/";
      homepage = "https://github.com/cocodataset/cocoapi";
      license = licenses.bsd0;
    };
  };

  detectron2 = (pkgs.python38Packages.buildPythonPackage rec {
    name = "detectron2";
    version = "0.2.1";
    src = builtins.fetchGit {
      url = "https://github.com/facebookresearch/detectron2.git";
      ref = "v0.2.1";
    };
    nativeBuildInputs = with pkgs; [
      which
    ];

    doCheck = false;

    propagatedBuildInputs = with pkgs.python38Packages; with pkgs; [
      cython
      pyyaml
      pytest
      pytestrunner
      pytorch
      pydot
      tqdm
      cloudpickle
      fvcore
      matplotlib
      pycocotools
      tensorflow-tensorboard
    ];
  });
  muscima = (pkgs.python38Packages.buildPythonPackage rec {
    name = "muscima";
    version = "0.10.0";
    src = builtins.fetchGit {
      url = "https://github.com/hajicj/muscima.git";
      ref = "master";
      rev = "f6f3d014761442af52a108bb873786a41d6de4b3";
    };
    checkPhase = "true"; # Skip checks 'cause they fail
    propagatedBuildInputs = with pkgs.python38Packages; [
      lxml
      numpy
      pytest
      pytestcov
      scikitimage
      scikitlearn
      matplotlib
      #midi2audio
      future
    ];
    removeBin = ''
      rm -r $out/bin
    '';
    postPhases = ["removeBin"];
  });
  mung = (pkgs.python38Packages.buildPythonPackage rec {
    name = "mung";
    version = "1.0";
    src = builtins.fetchGit {
      url = "https://github.com/OMR-Research/mung.git";
      ref = "master";
      rev = "9ff3e3addde3b200562d21c2b4990d3667398ee0";
    };
    checkPhase = "true"; # Skip checks 'cause they fail
    propagatedBuildInputs = with pkgs.python38Packages; [
      lxml
      numpy
      pytest
      pytestcov
      scikitimage
      scikitlearn
      matplotlib
      midiutil
      #midi2audio
      future
    ];
    removeBin = ''
      rm -r $out/bin
    '';
    postPhases = ["removeBin"];
  });
  omrdatasettools = (pkgs.python38Packages.buildPythonPackage rec {
    name = "omrdatasettools";
    version = "1.3.1";
    src = builtins.fetchGit {
      url = "https://github.com/apacha/omr-datasets.git";
      ref = "${version}";
    };
    checkPhase = "true"; # Skip checks 'cause they fail
    propagatedBuildInputs = with pkgs.python38Packages; [
      pillow
      pytest
      scikitimage
      h5py
      pyhamcrest
      muscima
      mung
      numpy
      lxml
      tqdm
      twine
      sympy
      sphinx_rtd_theme
      pandas
      ipython
    ];
  });
  
  pydeck = (pkgs.python38Packages.buildPythonPackage rec {
    pname = "pydeck";
    version = "0.6.1";

    src = pkgs.python3Packages.fetchPypi {
      inherit pname version;
      sha256 = "1l18iy3l6cgqlxwf9bvqijy494hx4hsnj1nm9i2pabz94i24hcd4";
    };

    doCheck = false; # I don't know how to tell it where to find pytest

    nativeBuildInputs = with pkgs.python38Packages; [
      pytest
    ];
    propagatedBuildInputs = with pkgs.python38Packages; [
      ipykernel
      ipywidgets
      traitlets
      jinja2
      numpy
    ];
  });


  my-python-packages = python-packages: with python-packages; [
    pip
    virtualenv
    pandas
    tqdm
    pillow
    requests
    scikitlearn
    detectron2
    pytorch
    torchvision
    opencv3
    omrdatasettools
  ];
  python-with-my-packages = pkgs.python38.withPackages my-python-packages;
in
  pkgs.mkShell {
    # nativeBuildInputs is usually what you want -- tools you need to run
    buildInputs = with pkgs; [
      python-with-my-packages
      ninja
      (streamlit.overridePythonAttrs (old: rec {
        version = "0.79.0";

        src = pkgs.python38Packages.fetchPypi {
          inherit version;
          inherit (old) pname format;
          sha256 = "085br2yi5l4xrb12bn8iykw10fmix5jy80ycl1s53qgynl0wkhim";
        };
        propagatedBuildInputs = (lib.remove pkgs.python38Packages.tornado_5 old.propagatedBuildInputs) ++ (with pkgs.python38Packages; [
          cachetools
          pyarrow
          pydeck
          GitPython
          tornado
        ]);
      }))
      opencv2
    ];
  }
