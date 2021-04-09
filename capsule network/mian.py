import os
import sys
import seaborn
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from config import cfg
from Cap_net import CapsNet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from Makeing_dataset import load_dataset
from spactial import load_spactial_dataset
from one import load_one_dataset
def save_to():
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        loss = cfg.results + '/loss.csv'
        train_acc = cfg.results + '/train_acc.csv'
        val_acc = cfg.results + '/val_acc.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return (fd_train_acc, fd_loss, fd_val_acc)
    else:
        test_acc = cfg.results + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return (fd_test_acc)



def plotHeatMap(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    # 归一化
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)
    ##S类
    FP = con_mat.sum(axis=0) - np.diag(con_mat)
    FN = con_mat.sum(axis=1) - np.diag(con_mat)
    TP = np.diag(con_mat)
    TN = con_mat.sum() - (FN + FP + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    print("FP", FP)
    print("FN", FN)
    print("TP", TP)
    print("TN", TN)
    each_acc = TP / (TP + FN)
    total_acc = np.diag(con_mat) / con_mat.sum()
    acc = (TP + TN) / (FP + FN + TP + TN)
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)
    ppr = TP / (TP + FP)
    f1 = 2 / (1 / sen + 1 / ppr)
    g = np.sqrt(sen * ppr)
    print("total_acc", total_acc)
    print("each_acc", each_acc)
    print("acc", acc)
    print("sen", sen)
    print("spe", spe)
    print("ppr", ppr)
    print("f1", f1)
    print("g", g)
    # 绘图
    plt.figure(figsize=(8, 8))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title("capsule network")
    plt.show()
    return

def train(model, supervisor, num_label):
    print("training data")
    #################################
    trX, trY, num_tr_batch , valX, valY,num_val_batch = load_dataset(cfg.batch_size, is_training=True)
    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    with supervisor.managed_session(config=config) as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print('Training for epoch ' + str(epoch) + '/' + str(cfg.epoch) + ':')
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=True, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_acc, summary_str ,margin_loss,reconstruction_loss= sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary,model.margin_loss,model.reconstruction_err])
                    print("margin_loss:",margin_loss)
                    print("reconstruction_loss:",reconstruction_loss)
                    print("total_loss:",loss)

                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op)

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:

                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        acc = sess.run(model.accuracy, {model.X: valX[start:end], model.labels: valY[start:end]})
                        val_acc += acc
                    val_acc = val_acc / (cfg.batch_size * num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()


def evaluation(model, supervisor, num_label):
    #######################################
    teX, teY, num_te_batch = load_dataset(cfg.batch_size, is_training=False)

    teY = teY[:cfg.batch_size*num_te_batch]
    fd_test_acc = save_to()
    Y_pre = []
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')
        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=True, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            Y_pre_batch, acc = sess.run([model.argmax_idx, model.accuracy], {model.X: teX[start:end], model.labels: teY[start:end]})

            Y_pre.append(Y_pre_batch)
            test_acc += acc
        Y_pre = np.array(Y_pre).reshape(-1,1)

        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        print("accuracy:",test_acc)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')
        plotHeatMap(teY, Y_pre)


def main(_):
    tf.logging.info(' Loading Graph...')
    ################################
    num_label = 5


    model = CapsNet()
    tf.logging.info(' Graph loaded')

    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

    if cfg.is_training:
        print("training")
        tf.logging.info(' Start training...')
        train(model, sv, num_label)
        tf.logging.info('Training done')
    else:
        print("test")
        evaluation(model, sv, num_label)

if __name__ == "__main__":
    tf.app.run()
