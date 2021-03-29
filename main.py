import os
import torch
import torchvision.transforms as T
import h5py
import random
# import numpy as np

from src import preproc as pp
from src.generator import Tokenizer, DataGenerator
from src.reader import Dataset

from network.model import make_model

from engine import LabelSmoothing, run_epochs
# get_memory


def preprocess_pages(pages, lines):
    if len(os.listdir(lines)) == 0:
        pp.preprocess(pages, img_size=(1024, 128, 1))
        print('Procesando paginas y lineas')
    else:
        print('Lineas procesadas')


def dataset_creator(raw, source_data):
    ds = Dataset(source=raw, name='persefone')
    ds.read_partitions()

    for i in ds.partitions:
        with h5py.File(source_data, "a") as hf:
            hf.create_dataset(f"{i}/dt", data=ds.dataset[i]['dt'], compression="gzip", compression_opts=4)
            hf.create_dataset(f"{i}/gt", data=ds.dataset[i]['gt'], compression="gzip", compression_opts=4)
            print(f"[OK] {i} partition.")

    print(f"Transformation finished.")


def train(device, source_path, charset_base, max_text_length, batch_size, lr, epochs, target_path):
    device = torch.device(device)
    model = make_model(tokenizer.vocab_size, hidden_dim=512, nheads=8,
                       num_encoder_layers=6, num_decoder_layers=6)
    model.to(device)
    transform = T.Compose([
        T.Resize((100, 1300)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = torch.utils.data.DataLoader(
        DataGenerator(source_path, charset_base, max_text_length, 'train', transform), batch_size=batch_size,
        shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        DataGenerator(source_path, charset_base, max_text_length, 'valid', transform), batch_size=batch_size,
        shuffle=False, num_workers=2)

    criterion = LabelSmoothing(size=tokenizer.vocab_size, padding_idx=0, smoothing=0.1)
    criterion.to(device)  # learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0004)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    run_epochs(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs, tokenizer, target_path, device)


def partitions_maker(transcriptions_path, partitions_path):
    files_lst = ['training.lst', 'validation.lst', 'test.lst']
    open(os.path.join(partitions_path, files_lst[0]), 'w').close()
    open(os.path.join(partitions_path, files_lst[1]), 'w').close()
    open(os.path.join(partitions_path, files_lst[2]), 'w').close()

    for transcript in os.listdir(transcriptions_path):
        n = random.choices([0, 1, 2], k=1, weights=[80, 12, 8])[0]
        written_transcript = transcript[:-4]
        verify_transcriptions(os.path.join(transcriptions_path, transcript))
        with open(os.path.join(partitions_path, files_lst[n]), 'a') as path:
            if os.stat(path.name).st_size == 0:
                path.write(written_transcript)
            else:
                path.write('\n' + written_transcript)


def verify_transcriptions(transcription):
    with open(transcription, 'r')as tr:
        text_transcript = tr.readlines()
        text_transcript = text_transcript[0]
    with open(transcription, 'w') as wr:
        wr.write(text_transcript.replace('\t', '    '))


if __name__ == '__main__':
    # If no pages are processed (then transformed in lines) then do it.
    # if it's already processed, then go to read the dataset
    raw_path = './data/'
    pages_path = os.path.join(raw_path, 'pages/')
    lines_path = os.path.join(raw_path, 'lines/')
    sources = os.path.join(raw_path, f"persefone.hdf5")
    output = os.path.join('.', '/output',)

    preprocess_pages(pages=pages_path, lines=lines_path)
    max_length = 128
    chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;?-_𓀀𓀁𓀂𓀃𓀄𓀅𓀆𓀇𓀈𓀉𓀊𓀋𓀌𓀍𓀎𓀏 '
    tokenizer = Tokenizer(chars=chars, max_text_length=max_length)
    # partitions_maker(transcriptions_path=os.path.join(raw_path, 'transcriptions/'),
    #                  partitions_path=os.path.join(raw_path, 'partitions/'))
    
    dataset_creator(raw=raw_path, source_data=sources)
    # train(device='cpu', source_path=sources, charset_base=chars,
    #       max_text_length=max_length, batch_size=16, lr=1e-4, epochs=200, target_path=output)
