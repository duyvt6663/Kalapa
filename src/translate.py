import html
import time
import random
import math

from googletrans import Translator


def translate_text(texts, dest="en", batch_size=16, delay_per_batch=2):
    translator = Translator()
    is_str = False
    if isinstance(texts, str):
        texts = [texts]
        is_str = True

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tries = 5
        while tries > 0:
            try:
                results.extend(translator.translate(batch, dest=dest))
                time.sleep(random.randint(0, delay_per_batch))
                break
            except Exception as e:
                tries -= 1
                print(e)
                print(
                    f"Retrying batch {i//batch_size}/{math.ceil(len(texts)/batch_size)}"
                )
                time.sleep(random.randint(0, delay_per_batch))

    results = [html.unescape(r.text) for r in results]
    if is_str:
        return results[0]
    else:
        return results
