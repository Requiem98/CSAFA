##############################################
#Default image configuration
##############################################

FROM visionlabsapienza/workgroup:container-Fix1-26112022

RUN pip3 install lightning && pip3 install jsonargparse[signatures]
