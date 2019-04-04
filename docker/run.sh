# THIS IS A COPY OF https://get.fenicsproject.org/ FOR REFERENCE



# #!/usr/bin/env bash
# #
# # This script downloads and installs the fenicsproject script providing
# # simplified use of FEniCS Docker images.

# # https://stackoverflow.com/questions/3466166/how-to-check-if-running-in-cygwin-mac-or-linux
# if [ "$(uname)" == "Darwin" ]; then
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
#     echo -e "Contact ${BLUE}fenics-support@googlegroups.com${NORMAL} for assistance."
# fi

# # Config
# URLBASE="https://get.fenicsproject.org"
# TMPDEST="/tmp/fenicsproject"
# # On MINGW64_NT /usr/local/bin does not exist and we cannot even create it!
# # However, $HOME/bin seems to work and be in the PATH by default on systems
# # that I have checked.  On Linux the vast majority of distributions now have
# # ~/.local/bin in the PATH by default.
# if [ "$OS" == "Linux" ]; then
#     DEST_DIR="$HOME/.local/bin"
# elif [ "$OS" == "Windows" ]; then
#     DEST_DIR="$HOME/bin"
# elif [ "$OS" == "Darwin" ]; then
#     DEST_DIR="/usr/local/bin"
# fi

# DEST="${DEST_DIR}/fenicsproject"

# # Setup
# RED="\033[1;31m"
# BLUE="\033[1;34m"
# GREEN="\033[1;32m"
# NORMAL="\033[0m"


# quickstart ()
# {
#     echo "To get started, run the command"
#     echo ""
#     echo -e "  ${BLUE}fenicsproject run${NORMAL}"
#     echo ""
#     echo -e "For more information, see ${BLUE}fenicsproject help${NORMAL}."
# }

# # Check if we have Docker
# TMP=$(docker -v)
# if [ ! $? -eq 0 ]; then
#     echo ""
#     OS=$(uname)
#     if [ "$OS"="Linux" ]; then
#         echo -e "It appears that ${RED}Docker is not installed${NORMAL} on your system."
#         echo ""
#         echo "Follow these instructions to install Docker, then come back and try again:"
#         echo ""
#         echo "  https://docs.docker.com/linux/step_one/"
#         echo ""
#     else
#         echo -e "It appears that ${RED}Docker is not installed${NORMAL} on your system."
#         echo ""
#         echo -e "Or you forgot to run this script in the ${GREEN}Docker Quickstart Terminal${NORMAL}."
#         echo "Follow these instructions to install Docker, then come back and try again:"
#         echo ""
#         echo "  https://www.docker.com/products/docker-toolbox"
#         echo ""
#     fi
#     exit 1
# fi

# # Download script
# rm -f $TMPDEST
# curl -s $URLBASE/fenicsproject > $TMPDEST

# # Check if user is in sudoers
# SUDO=""
# if [[ "$OS" == "Darwin" && ! -w "$DEST_DIR" ]]; then
#     echo -e "On macOS we need your sudo password to install a script into $DEST".
#     sudo -k
#     sudo -v
#     SUDO="sudo"
#     if [ ! $? -eq 0 ]; then
#         echo ""
#         echo -e "It appears that you are ${RED}not allowed to run sudo${NORMAL} on your system."
#         echo -e "You therefore need to manually copy the ${GREEN}fenicsproject${NORMAL} script to a"
#         echo -e "location of your choice and update your ${BLUE}PATH${NORMAL} environment variable"
#         echo -e "accordingly. The script is currently located at ${BLUE}${TMPDEST}${NORMAL}."
#         echo -e "When you have done this, follow the instructions below to get started."
#         echo ""
#         quickstart
#         exit
#     fi
# fi

# # Copy the script to PATH
# $SUDO mkdir -p $DEST_DIR
# $SUDO cp $TMPDEST $DEST
# $SUDO chmod a+rx $DEST
# echo -e "Successfully installed the ${GREEN}fenicsproject${NORMAL} script in ${DEST}."
# echo ""
# quickstart


# # docker create -ti -v ${PWD}:/home/fenics/shared -w /home/fenics/shared firemarmot:latest
# # docker container start goofy_lewin
# # docker container attach goofy_lewin
