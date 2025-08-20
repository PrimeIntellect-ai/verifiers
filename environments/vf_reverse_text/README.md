# vf-reverse-text

### Overview
- **Environment ID**: `vf-reverse-text`
- **Short description**: Reverse a given paragraph; evaluated by LCS similarity to the exact reversal.
- **Tags**: text, transformation, single-turn, xml

### Datasets
- **Primary dataset(s)**: `agentlans/wikipedia-paragraphs` mapped to question/answer pairs
- **Source links**: Hugging Face Datasets
- **Split sizes**: Train/eval split controlled by `num_train_examples` and `num_eval_examples`

### Task
- **Type**: single-turn
- **Parser**: `XMLParser(["think","answer"])`
- **Rubric overview**: LCS similarity between parsed answer and ground-truth reversed text; optional format check

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval vf-reverse-text
```

Configure model and sampling:

```bash
uv run vf-eval vf-reverse-text \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 2000, "num_eval_examples": 200}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Reports are written under `./environments/vf_reverse_text/reports/` and auto-embedded below.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `2000` | Number of training examples |
| `num_eval_examples` | int | `200` | Number of evaluation examples |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | LCS similarity between reversed text and parsed answer |
| `format_reward` | Adherence to `<think>`/`<answer>` XML format |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<details><summary>Reports</summary>
<details><summary>vf-reverse-text--v0.1.0--model=gpt-4.1--n=5--r=3--args=noargs</summary>
<p><a href="reports/vf-reverse-text--v0.1.0--model=gpt-4.1--n=5--r=3--args=noargs.html" target="_blank">Open full report</a></p>
<h3>vf-reverse-text: gpt-4.1 (n=5, r=3)</h3>
<div class="meta">
<div><b>Environment</b>: vf-reverse-text (v0.1.0)</div>
<div><b>Model</b>: <span class="code">gpt-4.1</span></div>
<div><b>Provider</b>: https://api.openai.com/v1</div>
<div><b>Samples</b>: n=5, r=3</div>
<div><b>Date</b>: 2025-08-11</div>
<div><b>Time</b>: 19:44:40</div>
<div><b>Sampling</b>: max_tokens=2048, temperature=0.7</div>
</div>

<h2>Reward</h2>
<table>
<tr><th>mean</th><th>std</th><th>n</th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th></tr>
<tr>
<td>0.9198</td>
<td>0.1572</td>
<td>15</td>
<td>0.6026</td>
<td>0.9012</td>
<td>0.96</td>
<td>1.0266</td>
<td>1.0709</td>
</tr>
</table>


<h2>Metrics</h2>
<table>
<tr>
<th>metric</th><th>mean</th><th>std</th><th>n</th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th>
</tr>

<tr>
<td>lcs_reward_func</td>
<td>0.7198</td>
<td>0.1572</td>
<td>15</td>
<td>0.4026</td>
<td>0.7012</td>
<td>0.76</td>
<td>0.8266</td>
<td>0.8709</td>
</tr>

<tr>
<td>format_reward_func</td>
<td>1.0</td>
<td>0.0</td>
<td>15</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
</tr>

</table>


<h2>Examples <span class="muted">(showing up to 15 of 15)</span></h2>
<div class="examples">
<table>
<tr><th>#</th><th>reward</th><th>lcs_reward_func</th><th>completion</th></tr>

<tr>
<td>0</td>
<td>0.9086</td>
<td>0.7086</td>
<td><pre>&lt;think&gt;
To reverse the given text, I will start from the last character and move backward to the first, ensuring that all punctuation, capitalization, and line breaks are preserved in reverse order.
&lt;/think&gt;
&lt;answer&gt;
.)ecnivorP nrehtuoS( agnenumihS dna idniiwL ,)ecnivorP nretsaE( alawcN dna abmaluK…</pre></td>
</tr>

<tr>
<td>1</td>
<td>0.9392</td>
<td>0.7392</td>
<td><pre>&lt;think&gt;
To reverse the given text, I need to take the entire passage and write it backward, starting from the last character to the first, preserving all punctuation and spacing, so that the text is a complete mirror image.
&lt;/think&gt;
&lt;answer&gt;
.yrotsih tnemanruot CCW ni laog ht000,1 eht gnidda adna a…</pre></td>
</tr>

<tr>
<td>2</td>
<td>1.0471</td>
<td>0.8471</td>
<td><pre>&lt;think&gt;
I will take the given text and reverse it character by character, as requested.
&lt;/think&gt;
&lt;answer&gt;
.shtnom wef a revo ecalp ekat ot sraeppa ti tub ,suougibma tahwemos si mlf eht fo emarfemit ehT .noituloveR eht fo ytilaer eht ot ecnedua eht esopxe ot noitarran s&#39;ogireS revo deyalp dna mlf eh…</pre></td>
</tr>

<tr>
<td>3</td>
<td>0.9703</td>
<td>0.7703</td>
<td><pre>&lt;think&gt;
First, I will take the entire provided text and reverse it character by character, ensuring that punctuation, quotations, and formatting are preserved exactly as in the original, only in reverse order.
&lt;/think&gt;
&lt;answer&gt;
.elb at dnuor a ni tsilepan a sa flesmih sraeppa seon seD &#34;.ytisned lai…</pre></td>
</tr>

<tr>
<td>4</td>
<td>0.96</td>
<td>0.76</td>
<td><pre>&lt;think&gt;
Reverse the provided text character by character, maintaining original punctuation and formatting.
&lt;/think&gt;
&lt;answer&gt;
.srotsevni nretseW truoc ot tfihs yciloP ngierof a fo trap erew seinapmoc esenihC dna ratauQ ,hsikruT ,naisalyaM tsniaga snoitca tnemnrevog lanoitisnart esenadnuS eht taht de…</pre></td>
</tr>

<tr>
<td>5</td>
<td>0.9658</td>
<td>0.7658</td>
<td><pre>&lt;think&gt;
To reverse the provided text, I need to flip the entire passage character by character, starting from the end and moving to the beginning, preserving all punctuation, capitalization, and spacing as in the original.
&lt;/think&gt;
&lt;answer&gt;
).ecnivorP nrehtuoS( agenunmihS dna idniiwL ,ecnivorP nret…</pre></td>
</tr>

<tr>
<td>6</td>
<td>1.095</td>
<td>0.895</td>
<td><pre>&lt;think&gt;
To reverse the given text, I will start from the end and work my way to the beginning, ensuring each sentence and word order is reversed accurately. Punctuation and formatting will be preserved as closely as possible in the reversed output.
&lt;/think&gt;
&lt;answer&gt;
.yrotsih tnemanruot CCW ni laog …</pre></td>
</tr>

<tr>
<td>7</td>
<td>0.4735</td>
<td>0.2735</td>
<td><pre>&lt;think&gt;
First, I will copy the given text. Then, I’ll reverse the entire text character by character, preserving formatting. The output will be an exact mirror, with the last character appearing first, and so on. Finally, I’ll paste the reversed text inside the &lt;answer&gt; tag.
&lt;/think&gt;
&lt;answer&gt;
.shtn…</pre></td>
</tr>

<tr>
<td>8</td>
<td>0.6579</td>
<td>0.4579</td>
<td><pre>&lt;think&gt;
To reverse the given text, I need to flip the entire passage character by character, so that the last character comes first, and so on. This includes all punctuation, spacing, and line breaks in the original order.
&lt;/think&gt;
&lt;answer&gt;
.elbmat dnuor a ni tsil enap a sa flesmih sraep pa seno se…</pre></td>
</tr>

<tr>
<td>9</td>
<td>1.0208</td>
<td>0.8208</td>
<td><pre>&lt;think&gt;
To reverse the given text, I need to flip the entire passage so that the last character comes first and the first character comes last, preserving all spaces and punctuation exactly as they appear, but in reverse order.
&lt;/think&gt;
&lt;answer&gt;
.srotsevni nretseW truoc ot tfi hs ycilop ngierof a f…</pre></td>
</tr>

<tr>
<td>10</td>
<td>0.9286</td>
<td>0.7286</td>
<td><pre>&lt;think&gt;
To reverse the given text, I need to flip the order of all characters from end to start, maintaining punctuation and spacing as they appear.
&lt;/think&gt;
&lt;answer&gt;
.)ecnivorP nrehtuoS( agnenumihS dna idniiwL ,)ecnivorP nretsaE( alawcN dna abmaluK ,)ecnivorP alupauL( okobmotuM ,)ecnivorP nretseW(…</pre></td>
</tr>

<tr>
<td>11</td>
<td>1.0324</td>
<td>0.8324</td>
<td><pre>&lt;think&gt;
To reverse the given text, I need to take the entire passage and flip it character by character, so the last character becomes the first, and so on.
&lt;/think&gt;
&lt;answer&gt;
.yrotsih tnemanruot CWV ni laog ht000,1 eht gnidda adna adnaB arabraB dna ,tnemanelruot eht ta laog tsetaf tsaf eht gnicrocs…</pre></td>
</tr>

<tr>
<td>12</td>
<td>1.0606</td>
<td>0.8606</td>
<td><pre>&lt;think&gt;
To reverse the given text, I will start from the last character and move to the first, preserving all punctuation, case, and formatting.
&lt;/think&gt;
&lt;answer&gt;
.shtnom wef a revo ecalp ekat ot sraeppa ti tub ,suougiubma tahwemos si mlf eht fo emarfemit ehT .noituloveR eht fo ytilaer eht ot ecnei…</pre></td>
</tr>

<tr>
<td>13</td>
<td>0.8937</td>
<td>0.6937</td>
<td><pre>&lt;think&gt;
To reverse the given text, I will start from the end and work towards the beginning, reversing the order of characters, including punctuation and spaces.
&lt;/think&gt;
&lt;answer&gt;
.elb at dnuor a ni tsil enap a sa flesmih sraep pa seno seD .ytisned laicos&#34; gnidda yb &#34;koob eht ni tcartsa llits dna ……</pre></td>
</tr>

<tr>
<td>14</td>
<td>0.8432</td>
<td>0.6432</td>
<td><pre>&lt;think&gt;
To reverse the given text, I will write each character in the opposite order, starting from the last character to the first, including all punctuation, spaces, and formatting.
&lt;/think&gt;
&lt;answer&gt;
.srotsevni nretseW truoc ot tfihs yciloP ngierof a fo trap erew seinapmoc esenihC dna ratauQ ,hsi…</pre></td>
</tr>

</table>
</div>
</details>
<details><summary>vf-reverse-text--v0.1.0--model=gpt-4.1-mini--n=5--r=3--args=noargs</summary>
<p><a href="reports/vf-reverse-text--v0.1.0--model=gpt-4.1-mini--n=5--r=3--args=noargs.html" target="_blank">Open full report</a></p>
<h3>vf-reverse-text: gpt-4.1-mini (n=5, r=3)</h3>
<div class="meta">
<div><b>Environment</b>: vf-reverse-text (v0.1.0)</div>
<div><b>Model</b>: <span class="code">gpt-4.1-mini</span></div>
<div><b>Provider</b>: https://api.openai.com/v1</div>
<div><b>Samples</b>: n=5, r=3</div>
<div><b>Date</b>: 2025-08-11</div>
<div><b>Time</b>: 19:43:01</div>
<div><b>Sampling</b>: max_tokens=2048, temperature=0.7</div>
</div>

<h2>Reward</h2>
<table>
<tr><th>mean</th><th>std</th><th>n</th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th></tr>
<tr>
<td>0.5033</td>
<td>0.27</td>
<td>15</td>
<td>0.16</td>
<td>0.1911</td>
<td>0.6048</td>
<td>0.7475</td>
<td>0.8502</td>
</tr>
</table>


<h2>Metrics</h2>
<table>
<tr>
<th>metric</th><th>mean</th><th>std</th><th>n</th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th>
</tr>

<tr>
<td>lcs_reward_func</td>
<td>0.3139</td>
<td>0.2567</td>
<td>15</td>
<td>0.0</td>
<td>0.0111</td>
<td>0.4048</td>
<td>0.5475</td>
<td>0.6502</td>
</tr>

<tr>
<td>format_reward_func</td>
<td>0.9467</td>
<td>0.0884</td>
<td>15</td>
<td>0.8</td>
<td>0.9</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
</tr>

</table>


<h2>Examples <span class="muted">(showing up to 15 of 15)</span></h2>
<div class="examples">
<table>
<tr><th>#</th><th>reward</th><th>lcs_reward_func</th><th>completion</th></tr>

<tr>
<td>0</td>
<td>0.8622</td>
<td>0.6622</td>
<td><pre>&lt;think&gt;
The task is to reverse the entire given text, character by character, preserving the spacing and punctuation as is.
&lt;/think&gt;
&lt;answer&gt;
.)ecinorP nrehtroN( anew&#39;Ng aP ayfesekuU ,)ecinorP agnihcuM( abminahkavniV ,)ecinorP latneC( ihsum akalwebiC ,)ecinorP nretsaW htroN( awkakluL adnubM ,)nrets…</pre></td>
</tr>

<tr>
<td>1</td>
<td>0.3464</td>
<td>0.1464</td>
<td><pre>&lt;think&gt;
To reverse the given text, I will start from the last sentence and proceed to the first, reversing the order of words and characters accordingly while maintaining the punctuation and coherence of the text.
&lt;/think&gt;
&lt;answer&gt;
.yratsih tnemnaruot CWW ni laog ht000,1 eht gnidda adna adnaB arabr…</pre></td>
</tr>

<tr>
<td>2</td>
<td>0.16</td>
<td>0.0</td>
<td><pre>&lt;think&gt;
I will reverse the entire given text character by character.
&lt;/think&gt;
&lt;answer&gt;
.hstnom wef a revo ecalp ekat ot srappa ti tub ,suoiugibma tnawemos si mlif eht fo emitemarfi ehT .noituloveR eht fo ytilaeR eht ot ecneiduA eht esopxe ot noitarran s&#39;oirgeS revo deyalp dna mlif eht otni detaropr…</pre></td>
</tr>

<tr>
<td>3</td>
<td>0.16</td>
<td>0.0</td>
<td><pre>&lt;think&gt;
The original text discusses the film adaptation of a novel and includes quotes and opinions from various people about the adaptation process and its success. To reverse the text, I will write the entire passage backward, starting from the last character to the first, maintaining the origina…</pre></td>
</tr>

<tr>
<td>4</td>
<td>0.7455</td>
<td>0.5455</td>
<td><pre>&lt;think&gt;
The task is to reverse the given text completely, character by character.
&lt;/think&gt;
&lt;answer&gt;
. srotsevni nretsaW truoc ot tfihs ycilop regnof a fo trap erew seinapmoc enihC dna ratahC ,kruT ,naisalaM tniaga snoitca s&#39;tnemnrevog lanoitartsnart esenesduS eht taht denipo deh mA misaN rotubirtno…</pre></td>
</tr>

<tr>
<td>5</td>
<td>0.8451</td>
<td>0.6451</td>
<td><pre>&lt;think&gt;
The task is to reverse the entire given text, maintaining the original punctuation and capitalization as they appear, but reversing the order of characters so the text reads backward.
&lt;/think&gt;
&lt;answer&gt;
.)ecinorP nrehtroN( anew&#39;Ng aP ayfesukU .)ecinorP agnihcuM( abminakahnkniV .)ecinorP lart…</pre></td>
</tr>

<tr>
<td>6</td>
<td>0.6048</td>
<td>0.4048</td>
<td><pre>&lt;think&gt;
To reverse the given text, I will write the entire paragraph backwards, starting from the last character to the first.
&lt;/think&gt;
&lt;answer&gt;
.yrotsih tnemuratroT CWW ni laog ht000,1 eht gnidda adna adnaB arabraB ,tnemuratrot eht ta laog tsetsaF eht gnirrocs abmeW omhsuL htiw ,raey tsrif rieht n…</pre></td>
</tr>

<tr>
<td>7</td>
<td>0.16</td>
<td>0.0</td>
<td><pre>&lt;think&gt;
The text discusses the film &#34;Memories of Underdevelopment,&#34; focusing on the character Sergio, a bourgeois writer who stays in Cuba. It mentions the historical context (Cuban Revolution, missile crisis), the narrative style (fragmented, subjective), and the film&#39;s portrayal of social changes…</pre></td>
</tr>

<tr>
<td>8</td>
<td>0.2223</td>
<td>0.0223</td>
<td><pre>&lt;think&gt;
I will reverse the entire text word by word, maintaining the punctuation associated with the original words.
&lt;/think&gt;
&lt;answer&gt;
.table round a in panelist a as himself appears Desnoes. density.&#34; social &#34;adding by book&#34; the in abstract still and … shapeless was that world a &#34;objectivized Alea…</pre></td>
</tr>

<tr>
<td>9</td>
<td>0.7496</td>
<td>0.5496</td>
<td><pre>&lt;think&gt;
To reverse the text, I will reverse the order of all characters in the provided passage, maintaining the same characters but in reverse sequence.
&lt;/think&gt;
&lt;answer&gt;
. srotsevni nretsE tnuoc ot tfihs ycilop rengof a fo trap erew seinapmoc enihC dna rhtaQ ,ksruT ,naisylanaM tniaga snoitca s&#39;tn…</pre></td>
</tr>

<tr>
<td>10</td>
<td>0.8172</td>
<td>0.6172</td>
<td><pre>&lt;think&gt;
The text lists various types of ceremonies in Zambia, distinguishing between disclosed and undisclosed ones, and mentions specific named ceremonies and their associated provinces. To reverse the text, I need to invert the order of all characters, including punctuation and spacing, from the …</pre></td>
</tr>

<tr>
<td>11</td>
<td>0.16</td>
<td>0.0</td>
<td><pre>&lt;think&gt;
To reverse the given text, I will start from the end and work my way to the beginning, reversing the order of characters in the entire text.
&lt;/think&gt;
&lt;answer&gt;
.yrotsih tnaruatrommuwC WWT ni laog ht000,1 eht gnidda adna adnaB arabraB ,tnemaruot eht ta laog tsetsaf eht gnirrocs abmewM omohsuL…</pre></td>
</tr>

<tr>
<td>12</td>
<td>0.6734</td>
<td>0.4734</td>
<td><pre>&lt;think&gt;
The task is to reverse the entire given text exactly as it is, including punctuation and capitalization.
&lt;/think&gt;
&lt;answer&gt;
.shtnom wef a revo ecalp ekat ot sraepi ti tub ,suoiuqibma tnawemos si mlif eht fo emitemarf ehT .noituloveR eht fo ytilaer eht ot ecneidua eht esopxe ot noitarran s&#39;oi…</pre></td>
</tr>

<tr>
<td>13</td>
<td>0.6077</td>
<td>0.4077</td>
<td><pre>&lt;think&gt;
I will reverse the entire text, including punctuation and capitalization, to create the reversed version of the original passage.
&lt;/think&gt;
&lt;answer&gt;
.talbat dnuor a ni tsilanap a sa flesmih sraepse senoDseD .&#34;ycnetsed laicos&#34; gnidda yb &#34;koob eht ni tcartsba llits dna …sselpahts saw taht dlro…</pre></td>
</tr>

<tr>
<td>14</td>
<td>0.4348</td>
<td>0.2348</td>
<td><pre>&lt;think&gt;
To reverse the given text, I will write the entire passage backwards, starting from the last character to the first. This includes reversing the order of words, letters, and punctuation marks so that the text reads completely backward.
&lt;/think&gt;
&lt;answer&gt;
. srotsevnI nrets eW truoc ot tfihs y…</pre></td>
</tr>

</table>
</div>
</details>
</details>
<!-- vf:end:reports -->
