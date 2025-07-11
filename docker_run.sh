echo off
echo "Building a docker image: (docker build --rm -t nlp .)"
echo "----------------------------------"
docker build --rm -t nlp .
echo "----------------------------------"
read -p "Press enter to continue"

echo "Starting the container using the docker image: (docker run -dp 8888:8888 --name=c-nlp nlp)"
echo "----------------------------------"
docker run -dp 8888:8888 --rm --name=c-nlp nlp
read -p "Container is running. Press enter to continue"
docker exec c-nlp jupyter server list
echo "----------------------------------"
echo "Please open click on the link above to open jupiter lab in browser"
echo "----------------------------------"
read -p "Press enter to continue"

echo "----------------------------------"
echo "Stop and delete the container (docker rm -f c-nlp)"
echo "----------------------------------"
docker rm -f c-nlp
echo "----------------------------------"

echo "Delete images (docker image prune and docker image rm nlp)"
read -p "Do you agree to delete those images: nlp? Y/n " a
echo "$a"
if [ $a == "Y" ] || [ $a == "y" ]; then
  docker image prune -f && docker image rm nlp
fi

echo "----------------------------------"
echo "You have got those images:"
docker images
echo "----------------------------------"
echo "-------------END------------------"
