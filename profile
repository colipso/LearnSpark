# /etc/profile: system-wide .profile file for the Bourne shell (sh(1))
# and Bourne compatible shells (bash(1), ksh(1), ash(1), ...).

if [ "$PS1" ]; then
  if [ "$BASH" ] && [ "$BASH" != "/bin/sh" ]; then
    # The file bash.bashrc already sets the default PS1.
    # PS1='\h:\w\$ '
    if [ -f /etc/bash.bashrc ]; then
      . /etc/bash.bashrc
    fi
  else
    if [ "`id -u`" -eq 0 ]; then
      PS1='# '
    else
      PS1='$ '
    fi
  fi
fi

# The default umask is now handled by pam_umask.
# See pam_umask(8) and /etc/login.defs.

if [ -d /etc/profile.d ]; then
  for i in /etc/profile.d/*.sh; do
    if [ -r $i ]; then
      . $i
    fi
  done
  unset i
fi

export JAVA_HOME=/home/hp/programmefiles/jdk1.8.0_73/
export PATH=$JAVA_HOME/bin:$PATH
export SCALA_HOME=/home/hp/programmefiles/scala-2.11.8/
export PATH=$SCALA_HOME/bin:$PATH
export SPARK_HOME=/home/hp/programmefiles/spark/
export PATH=$SPARK_HOME/bin:$PATH
export KIBANA_HOME=/home/hp/programmefiles/kibana/
export PATH=$KIBANA_HOME/bin:$PATH
export ELASTICSEARCH_HOME=/home/hp/programmefiles/elasticsearch/
export PATH=$ELASTICSEARCH_HOME/bin:$PATH
export NODEJS_HOME=/home/hp/programmefiles/node/
export PATH=$NODEJS_HOME/bin:$PATH

