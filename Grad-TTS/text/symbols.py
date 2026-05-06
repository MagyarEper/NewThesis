""" from https://github.com/keithito/tacotron — modified for Hungarian (v3) """

_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_hungarian = 'áéíóöőúüűÁÉÍÓÖŐÚÜŰ'

# Hungarian grapheme vocabulary — no ARPAbet, no CMU dictionary needed
# vocab size: 82 (was 148 with ARPAbet)
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_hungarian)
