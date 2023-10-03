##############################################
#Default image configuration
##############################################

FROM visionlabsapienza/workgroup:container-Fix1-26112022

RUN pip3 install lightning && pip3 install jsonargparse[signatures]

RUN apt update && apt install -y git

WORKDIR /home

COPY getGitHubRepo.sh ./

RUN chmod 755 ./getGitHubRepo.sh

WORKDIR /home/GeoLoc

#COPY ./Data/small_CVUSA ./Data/small_CVUSA
COPY ./Data/University-Release ./Data/University-Release

ENTRYPOINT ["/home/getGitHubRepo.sh"]

