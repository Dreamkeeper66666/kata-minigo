Kata-MiniGo: Load and run KataGo's TensorFlow weights with MiniGo GTP
==================================================


Playing Against Kata-MiniGo
----------------------

Download the latest KataGo TF model at https://katagotraining.org/networks/

```shell
# Latest model should look like: /path/to/models/kata1-b40c256-sxxx-dxxx/saved_model/
LATEST_MODEL=/path/to/models/kata1-b40c256-sxxx-dxxx/saved_model/
READOUTS=1600
python kata_gtp.py --save_dir=$LATEST_MODEL --num_readouts=$READOUTS --verbose=3 --name_scope=swa_model
```

After some loading messages, it will display `GTP engine ready`, at which point
it can receive commands.  GTP cheatsheet:

```
genmove [color]             # Asks the engine to generate a move for a side
play [color] [coordinate]   # Tells the engine that a move should be played for `color` at `coordinate`
showboard                   # Asks the engine to print the board.
```

One way to play via GTP is to use gogui-display (which implements a UI that
speaks GTP.) You can download the gogui set of tools at
[http://gogui.sourceforge.net/](http://gogui.sourceforge.net/). See also
[documentation on interesting ways to use
GTP](http://gogui.sourceforge.net/doc/reference-twogtp.html).

```shell
gogui-twogtp -black 'python3 gtp.py --save_dir=$LATEST_MODEL --name_scope=swa_model' -white 'gogui-display' -size 19 -komi 7.5 -verbose -auto
```

Another way to play via GTP is to watch it play against GnuGo, while
spectating the games:

```shell
BLACK="gnugo --mode gtp"
WHITE="python3 gtp.py --load_file=$LATEST_MODEL"
TWOGTP="gogui-twogtp -black \"$BLACK\" -white \"$WHITE\" -games 10 \
  -size 19 -alternate -sgffile gnugo"
gogui -size 19 -program "$TWOGTP" -computer-both -auto
```

Running Minigo on a Kubernetes Cluster
==============================

See more at [cluster/README.md](https://github.com/tensorflow/minigo/tree/master/cluster/README.md)
