# This file contains
# detectron2, yacs, iopath, fvcore
# pycocotools
# muscima, mung, omrdatasettools
# pydeck
let
  # Pin nixpkgs

  # Some nixpkgs with open pull requests
  pkgsOMR = builtins.fetchTarball {
    name = "nixpkgs-omr-2021-04-24";
    url = "https://github.com/piegamesde/nixpkgs/archive/7e5f1ffcb8a777c86d209dcda7ddd67a7b5a9745.tar.gz";
    sha256 = "1ipr5hjlvazrq9ycxc0ggm8hpgzpmp6vv6pghankszsmv6n28w55";
  };

  pkgsDetectron = builtins.fetchTarball {
    name = "nixpkgs-detectron-2021-04-24";
    url = "https://github.com/piegamesde/nixpkgs/archive/df1c73939db02fdb80871aa3f26255ffa195787e.tar.gz";
    sha256 = "0p7jz832yaf8wjgqg2lns9xhnjrzxz2z6q398zcgyj5g33zlq1cl";
  };

  # The "main" nixpkgs
  pkgs = import (builtins.fetchTarball {
    name = "nixpkgs-unstable-2021-04-24";
    url = "https://github.com/nixos/nixpkgs/archive/abd57b544e59b54a24f930899329508aa3ec3b17.tar.gz";
    sha256 = "0d6f0d4j5jhnvwdbsgddc62qls7yw1l916mmfq5an9pz5ykc9nwy";
  }) {
    overlays = [
      (pkgs: super: {
        python38 = super.python38.override {
          # Careful, we're using a different self and super here!
          packageOverrides = pkgs: super: {
            # dask = pkgs.callPackage "${pkgsDetectron}/pkgs/development/python-modules/dask/default.nix" { };
            dask = super.dask.overridePythonAttrs (old: {doCheck = false;});
            
            iopath = pkgs.callPackage "${pkgsDetectron}/pkgs/development/python-modules/iopath/default.nix" { };
            yacs = pkgs.callPackage "${pkgsDetectron}/pkgs/development/python-modules/yacs/default.nix" { };
            fvcore = pkgs.callPackage "${pkgsDetectron}/pkgs/development/python-modules/fvcore/default.nix" { };
            detectron2 = pkgs.callPackage "${pkgsDetectron}/pkgs/development/python-modules/detectron2/default.nix" { };

            mung = pkgs.callPackage "${pkgsOMR}/pkgs/development/python-modules/mung/default.nix" { };
            muscima = pkgs.callPackage "${pkgsOMR}/pkgs/development/python-modules/muscima/default.nix" { };
            omrdatasettools = pkgs.callPackage "${pkgsOMR}/pkgs/development/python-modules/omrdatasettools/default.nix" { };
            
            pydeck = (pkgs.buildPythonPackage rec {
              pname = "pydeck";
              version = "0.6.1";

              src = pkgs.fetchPypi {
                inherit pname version;
                sha256 = "1l18iy3l6cgqlxwf9bvqijy494hx4hsnj1nm9i2pabz94i24hcd4";
              };

              checkInputs = with pkgs; [
                jupyter
                pandas
                pytestCheckHook
              ];

              disabledTests = [
                "test_nbconvert" # Does internet
              ];

              propagatedBuildInputs = with pkgs; [
                ipykernel
                ipywidgets
                traitlets
                jinja2
                numpy
              ];
            });
            
          };
        };
        python38Packages = super.recurseIntoAttrs (pkgs.python38.pkgs);

        streamlit = (super.streamlit.overridePythonAttrs (old: rec {
          version = "0.79.0";

          src = pkgs.python38Packages.fetchPypi {
            inherit version;
            inherit (old) pname format;
            sha256 = "085br2yi5l4xrb12bn8iykw10fmix5jy80ycl1s53qgynl0wkhim";
          };
          propagatedBuildInputs = (pkgs.lib.remove pkgs.python38Packages.tornado_5 old.propagatedBuildInputs) ++ (with pkgs.python38Packages; [
            cachetools
            pyarrow
            pydeck
            GitPython
            tornado
          ]);
        }));

      })
    ];
  };

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
      streamlit
      opencv2
    ];
  }
