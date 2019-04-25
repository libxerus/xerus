# THIS IS A COPY OF https://get.fenicsproject.org/fenicsproject FOR REFERENCE



# #!/usr/bin/env bash
# #
# # This script wraps Docker commands to simplify the use of FEniCS Docker images.
# # Authors: Jack S. Hale <jack.hale@uni.lu>
# #          Anders Logg <logg@simula.no>

# # Config
# DEFAULT_IMAGE_HOST="quay.io/fenicsproject"
# DEFAULT_IMAGE="stable"
# DEFAULT_MAKEFLAGS="-j1"
# # Alternatively, to detect:
# #DEFAULT_MAKEFLAGS=$(grep -c '^processor' /proc/cpuinfo)
# # Where to mount the Instant/dijitso cache 
# # in the container.
# CACHE_DIR_CONTAINER="/home/fenics/.cache/fenics"
# # Where to build FEniCS.
# BUILD_DIR_CONTAINER="/home/fenics/local"
# # Where to store the FEniCS source code.
# SRC_DIR_CONTAINER="/home/fenics/local/src"
# # Share in the current working directory
# PROJECT_DIR_HOST="'$(pwd)'"
# # to the following directory in the container.
# PROJECT_DIR_CONTAINER="/home/fenics/shared"
# # Workaround for issue https://github.com/docker/docker/issues/9299
# DEFAULT_COMMAND='/bin/bash -l -c "export TERM=xterm; bash -i"'

# BIND_PORT_RANGE="3000-4000"
# DOCKER_NB_PORT="8888"
# DOCKER_MPL_PORT="8000"

# # Setup
# set -e
# RED="\033[1;31m"
# GREEN="\033[1;32m"
# BLUE="\033[1;34m"
# NORMAL="\033[0m"

# # Detect OS
# # https://stackoverflow.com/questions/3466166/how-to-check-if-running-in-cygwin-mac-or-linux
# if [[ "$(docker info | grep -c 'Docker for Mac')" == "1" || "$(docker info | grep -c Alpine)" == "1" ]]; then
#     OS="Docker"
# elif [ "$(uname)" == "Darwin" ]; then
#     # Mac OS X with Docker installed using docker-machine.
#     OS="Darwin"
# elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
#     # If we are running locally on Linux.
#     OS="Linux"
# elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
#     # The Docker environment installed by Docker Toolbox is MinGW based.
#     # Docker does not run on 32-bit environments.
#     # Don't know if we want to support MSYS or Cygwin as well.
#     OS="Windows"
# else
#     echo -e "${RED}Error${NORMAL}: We do not currently support your operating system $(uname)."
#     echo "Contact ${BLUE}fenics-support@googlegroups.com${NORMAL} for assistance."
# fi

# # Linux users have an issue with sharing files because the UID and GID in the
# # container sometimes does not match the UID and GID on the host. By passing
# # the UID and GID on the host, the container will modify the fenics user to
# # match the user on the host, resulting in seamless sharing.
# # Only use this trick on Linux, reduces complexity and not needed on Windows
# # and Mac hosts. Problems have been reported using Docker Toolbox for Mac
# # that were fixed when not using the UID/GID mapping.
# CHANGE_UID_GID=""
# if [ "$OS" == "Linux" ]; then
#     HOST_UID=$(id -u)
#     HOST_GID=$(id -g)
#     CHANGE_UID_GID="--env HOST_UID=${HOST_UID} --env HOST_GID=${HOST_GID}"
# fi

# # For security: 
# # On Windows and Mac with docker-machine we bind to the Virtualbox IP. Only the
# # host machine can see this IP.
# # On Docker for Mac or Docker for Windows we bind
# # to the default private IP only visible to the host.
# # On Linux we bind only on localhost. Only the host machine
# # can see this IP.
# if [[ "$OS" == "Darwin" || "$OS" == "Windows" ]]; then
#     # Ask docker-machine for the IP address of the VM 
#     ACTIVE_MACHINE=$(docker-machine active)
#     DOCKER_BIND_IP=$(docker-machine ip $ACTIVE_MACHINE)
# elif [[ "$OS" == "Docker" ]]; then
#     # Will bind to private IP by default.
#     # Can we work out what this IP is? Specific is more secure.
#     DOCKER_BIND_IP="127.0.0.1"
# else
#     # Conservative default, works on Linux.
#     DOCKER_BIND_IP="127.0.0.1"
# fi

# BIND_NB="-p ${DOCKER_BIND_IP}:${BIND_PORT_RANGE}:${DOCKER_NB_PORT}"
# BIND_MPL="-p ${DOCKER_BIND_IP}:${BIND_PORT_RANGE}:${DOCKER_MPL_PORT}"

# help ()
# {
#     echo "Usage: fenicsproject <command> [options]"
#     echo ""
#     echo "  fenicsproject run [image] [command]   - run a simple FEniCS session."
#     echo "  fenicsproject create <name> [image]   - create standard project with given name."
#     echo "  fenicsproject notebook <name> [image] - create notebook project with given name."
#     echo "  fenicsproject start <name>            - start session in project with given name."
#     echo "  fenicsproject pull <image>            - pull latest version of given image."
#     echo "  fenicsproject clean-cache [image]     - clean the shared FEniCS (Instant) cache."

#     echo ""
#     echo "Use 'fenicsproject run' for simple (non-persistent) FEniCS sessions."
#     echo ""
#     echo "For persistent sessions, use 'create' or 'notebook' followed by 'start'"
#     echo "to (re)start a session."
#     echo ""
#     echo "Available images:"
#     echo ""
#     echo "  stable  - latest stable release [default]"
#     echo "  dev     - latest development version, master branch"
#     echo "  dev-env - development environment including dependencies but not FEniCS"
#     echo ""
#     echo "You can update to the latest version of this script at any time using:"
#     echo ""
#     echo "  curl -s https://get.fenicsproject.org | bash"
#     echo ""
#     echo "For more details and tips, see our FEniCS Docker page:"
#     echo ""
#     echo "  http://fenics-containers.readthedocs.org/en/latest/"
#     echo ""
# }

# command ()
# {
#     echo "[$@]"
#     echo ""
#     eval $@
# }

# command-swallow-stderr ()
# {
#     echo "[$@]"
#     echo ""
#     eval $@ 2> /dev/null
# }

# clean-string () {
#     # Docker cannot name any objects (volumes, containers)
#     # with strings containing / or :. As we use the image
#     # name as a (somewhat) unique identifier to create
#     # various caches, strip these components out.
#     local IMAGE="$1"
#     # Remove slashes
#     IMAGE=${IMAGE////}
#     # Remove colons
#     IMAGE=${IMAGE/:/}
#     # Remove ampersands
#     IMAGE=${IMAGE/@/}
#     echo $IMAGE
# }

# get_ip_port () {
#     # get host ip, port for connecting to a docker ip, port
#     NAME=$1
#     PORT=$2

#     HOST_PORT=$(docker port $NAME $PORT | cut -d: -f2)

#     if [[ "$OS" == "Darwin" || "$OS" == "Windows" ]]; then
#         # Ask docker-machine for the IP address of the instance
#         ACTIVE_MACHINE=$(docker-machine active)
#         IP=$(docker-machine ip $ACTIVE_MACHINE)
#     elif [[ "$OS" == "Docker" ]]; then
#         IP=$(docker inspect --format="{{ ( index (index .NetworkSettings.Ports \"$PORT/tcp\") 0).HostIp }}" $NAME 2>/dev/null)
#         if [[ "$IP" = "0.0.0.0" ]]; then
#             # 0.0.0.0 isn't a valid connnect, so use 127
#             IP="127.0.0.1"
#         fi
#     else
#         IP="localhost"
#     fi
#     echo $IP $HOST_PORT
# }

# log_mpl_port () {
#     read IP MPL_PORT < <(get_ip_port $1 ${DOCKER_MPL_PORT})
#     echo -e "${BLUE}After calling 'plt.show()' you can access matplotlib plots at http://${IP}:${MPL_PORT}${NORMAL}"
# }

# log_nb_port () {
#     read IP NB_PORT < <(get_ip_port $1 ${DOCKER_NB_PORT})
#     echo -e "${BLUE}You can access the Jupyter notebook at http://${IP}:${NB_PORT}${NORMAL}"
# }

# run ()
# {
#     IMAGE="$1"
#     COMMAND="$2"
#     # Setup a container for instant cache
#     create-cache-container $IMAGE

#     CMD="docker create -ti \
#            ${BIND_MPL} \
#            ${CHANGE_UID_GID} \
#            --env MAKEFLAGS=${DEFAULT_MAKEFLAGS} \
#            -v instant-cache-$(clean-string $IMAGE):$CACHE_DIR_CONTAINER \
#            --env INSTANT_CACHE_DIR=$CACHE_DIR_CONTAINER/instant \
#            --env DIJITSO_CACHE_DIR=$CACHE_DIR_CONTAINER/dijitso \
#            -v $PROJECT_DIR_HOST:$PROJECT_DIR_CONTAINER \
#            -w $PROJECT_DIR_CONTAINER \
#            --label org.fenicsproject.created_by_script=true \
#            $IMAGE '$COMMAND'"

#     echo "[$CMD]" | tr -s " "; echo ""
#     NAME=$(command $CMD | tail -n 1)
#     command docker start $NAME

#     log_mpl_port $NAME

#     command docker attach $NAME
#     command docker rm $NAME
# }

# create-cache-container ()
# {
#     IMAGE="$1"
#     CMD="docker volume create --name instant-cache-$(clean-string $IMAGE)"
#     command $CMD
# }

# create-build-container ()
# {
#     NAME="$1"
#     CMD="docker volume create --name fenics-build-${NAME}"
#     command $CMD
# }

# clean-cache ()
# {
#     IMAGE="$1"
#     CMD="docker run --rm -v instant-cache-$(clean-string $IMAGE):$CACHE_DIR_CONTAINER $IMAGE 'rm -rf $CACHE_DIR_CONTAINER/*'"
#     command $CMD
# }

# create ()
# {
#     NAME="$1"
#     IMAGE="$2"

#     echo -e "Creating new ${GREEN}FEniCS Project${NORMAL} terminal project ${BLUE}$NAME${NORMAL}."
#     echo ""

#     # Share FENICS_SRC_DIR if set
#     if [ ! -z "$FENICS_SRC_DIR" ]; then
#         SHARE_SRC_DIR="-v '$FENICS_SRC_DIR':$SRC_DIR_CONTAINER"
#     fi

#     # In the case that a container is created from a dev-env image, we store
#     # the compiled FEniCS library inside a Docker volume.  This will allow us
#     # to share compiled FEniCS versions between containers by mounting the
#     # volume read-only in a new dev-env container.
#     if [[ $IMAGE == *dev-env ]]; then
#        create-build-container $NAME
#        VOLUME_BUILD_DIR="-v fenics-build-${NAME}:$BUILD_DIR_CONTAINER" 
#     fi

#     # People using dev-env-dbg might want to use gdb. Because docker
#     # has very strict isolation, we need to add some capabilities
#     # into the container to allow gdb to work.
#     if [[ $IMAGE == *dev-env-dbg ]]; then
#         # First option is less than ideal security wise.
#         # should be improved.
#         CAPABILITIES="--security-opt=seccomp=unconfined --cap-add=SYS_PTRACE"
#     fi

#     # Create a shared container to hold the instant cache
#     create-cache-container $IMAGE
#     CMD="docker create -ti --name $NAME \
#            ${BIND_NB} ${BIND_MPL}\
#            ${CHANGE_UID_GID} \
#            --env MAKEFLAGS=${DEFAULT_MAKEFLAGS} \
#            -v instant-cache-$(clean-string $IMAGE):$CACHE_DIR_CONTAINER \
#            --env INSTANT_CACHE_DIR=$CACHE_DIR_CONTAINER/instant \
#            --env DIJITSO_CACHE_DIR=$CACHE_DIR_CONTAINER/dijitso \
#            --label org.fenicsproject.created_by_script=true \
#            --label org.fenicsproject.project_type='standard' \
#            -v $PROJECT_DIR_HOST:$PROJECT_DIR_CONTAINER \
#            $SHARE_SRC_DIR \
#            $VOLUME_BUILD_DIR \
#            $CAPABILITIES \
#            -w $PROJECT_DIR_CONTAINER \
#            $IMAGE $DEFAULT_COMMAND"
#     command $CMD

#     # Print summary
#     echo ""
#     echo -e "To ${BLUE}start the session${NORMAL}, type the following command:"
#     echo ""
#     echo "  fenicsproject start $NAME"
#     echo ""
#     echo "You will find the current working directory $PROJECT_DIR_HOST under ~/shared."
# }

# start ()
# {
#     NAME="$1"

#     # Determine whether container is already running
#     # If not, then start a new session (bash or notebook).
#     # If it is, then launch a new bash login terminal.
#     IS_RUNNING=$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)
#     PROJECT_TYPE=$(docker inspect -f '{{index .Config.Labels "org.fenicsproject.project_type" }}' $NAME 2>/dev/null)
#     if [ "$IS_RUNNING" == "false" ]; then
#         echo "Starting project named $NAME."
#         CMD="docker start $NAME"
#         command $CMD

#         log_mpl_port $NAME

#         if [ "$PROJECT_TYPE" == "notebook" ]; then
#             log_nb_port $NAME
#         fi
#         CMD="docker attach $NAME"
#         command $CMD
#     else
#         echo "Starting new session in project named $NAME."
#         CMD="docker exec -u fenics -ti $NAME $DEFAULT_COMMAND"
#         command $CMD
#     fi
# }

# pull ()
# {
#     IMAGE="$1"
#     CMD="docker pull $IMAGE"
#     command $CMD
# }

# notebook ()
# {
#     IMAGE="$2"

#     NAME="$1"
#     echo -e "Creating new ${GREEN}FEniCS Project${NORMAL} notebook project ${BLUE}$NAME${NORMAL}."
#     echo ""

#     # Setup a container for instant cache
#     create-cache-container $IMAGE

#     CMD="docker create \
#            ${BIND_NB} ${BIND_MPL}\
#            ${CHANGE_UID_GID} \
#            --env MAKEFLAGS=${DEFAULT_MAKEFLAGS} \
#            -v instant-cache-$(clean-string $IMAGE):$CACHE_DIR_CONTAINER \
#            --env INSTANT_CACHE_DIR=$CACHE_DIR_CONTAINER/instant \
#            --env DIJITSO_CACHE_DIR=$CACHE_DIR_CONTAINER/dijitso \
#            -v $PROJECT_DIR_HOST:$PROJECT_DIR_CONTAINER \
#            -w /home/fenics \
#            --label org.fenicsproject.created_by_script=true \
#            --label org.fenicsproject.project_type='notebook' \
#            --name $NAME \
#            $IMAGE 'jupyter-notebook --ip=0.0.0.0'"
#     command $CMD

#     # Print summary
#     echo ""
#     echo -e "To ${BLUE}start the session${NORMAL}, type the following command:"
#     echo ""
#     echo "  fenicsproject start $NAME"
#     echo ""
#     echo "You will find the current working directory $PROJECT_DIR_HOST under ~/shared."
# }

# check_name () {
#     NAME=$1
#     if [ -z "$NAME" ]; then
#         echo -e "${RED}Error${NORMAL}: You must specify the name of the project you want to start."
#         exit 1
#     fi
#     set +e
#     IS_RUNNING=$(docker inspect -f '{{.State.Running}}' $NAME 2> /dev/null) 
#     if [ $? -eq 1 ]; then
#         echo -e "${RED}Error${NORMAL}: Project $NAME does not exist."
#         echo "You can create a project with the command:"
#         echo ""
#         echo "    fenicsproject create $NAME"
#         echo ""
#         echo "and then try running this command again." 
#         exit 1 
#     fi
#     set -e
# }

# fail_if_project_already_exists () {
#     NAME=$1
#     set +e
#     IS_RUNNING=$(docker inspect -f '{{.State.Running}}' $NAME 2> /dev/null)
#     if [ $? -eq 0 ]; then
#         echo -e "${RED}Error${NORMAL}: Project $NAME already exists!"
#         echo ""
#         echo -e "You can try a different name, or ${BLUE}permanently${NORMAL} delete the existing project"
#         echo "with the command:"
#         echo ""
#         echo "    docker rm $NAME"
#         echo ""
#         echo -e "Files in the folder ~/shared will ${BLUE}not${NORMAL} be deleted."
#         exit 1
#     fi
#     set -e
# }

# preprocess_image_name () {
#     IMAGE=$1
#     # Check if we have a fully qualified image name, e.g. quay.io/dolfinadjoint/dolfin-adjoint
#     if [[ "$IMAGE" == *\/* ]]; then
#         :
#         # Do nothing, user passing their own fully qualified image name and we assume
#         # they know what they are doing. 
#     else
#         # Otherwise, they are passing an abbreviated name, e.g. stable.
#         # so we need to make it a fully qualified name.
#         if [ "$DEFAULT_IMAGE_HOST" == "quay.io/fenicsproject" ]; then
#             fail=true
#             # List should contain all of our official images suitable for end-users.
#             # add : to allow users to specify tags
#             for image in 'stable' 'dev' 'dev-env'; do
#                 if [[ "$IMAGE:" == "$image:"* ]]; then
#                    fail=false
#                 fi
#             done
#             # We keep a tag stable:current so we can shift stable conservatively.
#             # All other images shift with default tag :latest.
#             if [ "$IMAGE" == "stable" ]; then
#                 IMAGE=${IMAGE}":current" 
#             fi
#             if [ $fail == true ]; then
#                 echo -e "${RED}Error${NORMAL}: Image with name $IMAGE does not exist. Try stable, dev-env or dev."
#                 exit 1
#             fi
#         fi
#         # Prepend DEFAULT_IMAGE_HOST so we have a fully qualified name.
#         IMAGE=$DEFAULT_IMAGE_HOST"/"$IMAGE 
#     fi
# }

# fail_if_home_directory ()
# {
#     if [ "$(pwd)" == $HOME ]; then
#         echo -e "${RED}Error${NORMAL}: We strongly advise against sharing your entire home directory"
#         echo "into a container. Instead, make a logical folder for each project:"
#         echo ""
#         echo "    mkdir ${HOME}/my-project"
#         echo ""
#         echo -e "and then run the ${BLUE}fenicsproject${NORMAL} script there:"
#         echo ""
#         echo "    cd ${HOME}/my-project"
#         echo "    fenicsproject $@"
#         exit 1
#     fi
# }

# # Check command-line arguments
# if [ "$1" == "run" ]; then
#     IMAGE="$2" 
#     : ${IMAGE:="$DEFAULT_IMAGE"}
#     preprocess_image_name $IMAGE
#     fail_if_home_directory
#     # Select command (if any)
#     if [ $# -ge 3 ]; then
#         shift; shift;
#         COMMAND="$@"
#     else
#         COMMAND=$DEFAULT_COMMAND
#     fi
#     run $IMAGE "$COMMAND"
# elif [ "$1" == "create" ]; then
#     # Select image
#     NAME="$2"
#     IMAGE="$3" 
#     : ${IMAGE:="$DEFAULT_IMAGE"}
#     preprocess_image_name $IMAGE
#     fail_if_project_already_exists $NAME
#     fail_if_home_directory
#     create $NAME $IMAGE
# elif [ "$1" == "notebook" ]; then
#     NAME="$2"
#     IMAGE="$3" 
#     : ${IMAGE:="$DEFAULT_IMAGE"}
#     preprocess_image_name $IMAGE
#     fail_if_project_already_exists $NAME
#     fail_if_home_directory
#     notebook $NAME $IMAGE
# elif [ "$1" == "start" ]; then
#     NAME="$2"
#     check_name $NAME 
#     start $NAME
# elif [ "$1" == "pull" ]; then
#     IMAGE="$2" 
#     : ${IMAGE:="$DEFAULT_IMAGE"}
#     preprocess_image_name $IMAGE
#     pull $IMAGE
# elif [ "$1" == "clean-cache" ]; then
#     IMAGE="$2" 
#     : ${IMAGE:="$DEFAULT_IMAGE"}
#     preprocess_image_name $IMAGE
#     clean-cache $IMAGE
# else
#     help
#     exit 1
# fi
