{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  buildInputs = [
    pkgs.gcc
    pkgs.cmake

    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.numpy
    pkgs.python311Packages.virtualenv
  ];

  LD_LIBRARY_PATH = "${pkgs.gcc.cc.lib}/lib:${pkgs.stdenv.cc.cc.lib}/lib";
}
