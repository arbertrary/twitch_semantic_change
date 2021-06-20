# Building on vingilot with buildah

export NAME=lsx-harbor.informatik.uni-wuerzburg.de/containers/cuda:pytorch
buildah bud -t $NAME -f CUDADOCKERFILE .
buildah push $NAME
