# Custom act runner image based on catthehacker/ubuntu:act-latest.
# Replaces the /var/run -> /run symlink with a real directory so that
# Docker 27+ doesn't reject act's toolset copy with "path escapes from parent".
FROM catthehacker/ubuntu:act-latest
RUN rm /var/run && mkdir -p /var/run
