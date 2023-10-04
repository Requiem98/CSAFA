#!/bin/sh
git clone https://github.com/Requiem98/GeoLoc.git /home/temp

mv /home/temp/* ./

#Exect next command
exec "$@"