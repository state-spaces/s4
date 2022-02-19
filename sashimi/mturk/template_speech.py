
TEMPL = """
<style>
th, td {{
    border: 2px solid black;
    padding: 8px;
}}

th {{
    width: 100px;
    vertical-align: text-top;
    font-weight: normal;
}}

.noborder {{
    border: 0px solid black;
}}

.thlab {{
    margin-bottom: 1em;
}}

.td {{
    text-align: center;
    vertical-align: middle;
}}

input[type=radio] {{
    border: 0px;
    width: 100%;
    height: 4em;
}}

audio {{
    width: 300px;
}}

input[type=submit] {{
    margin-top: 20px;
    width: 20em;
    height: 2em;
}}

</style>

<html>

<h2>Rate and annotate audio files containing spoken digits.</h2>

<p><b>Please use headphones in a quiet environment if possible. Read the instructions below carefully before starting the task.</b></p>
<p>You are presented a batch of recordings and asked to classify what digit you hear in each of them. If you are unsure which digit it is, select the one that sounds most like the recording to you.</p>
<p>You are also asked to rate the intelligibility of each recording</p>
<p><b>Intelligibility:</b> How easily could you identify the recorded digits? Are they impossible to classify (not at all intelligible) or very easy to understand (extremely intelligible)?</p>
<p>At the bottom, you'll be asked to provide your opinion of the recordings.</p>
<p><b>Think about the recordings you heard when answering these questions.</b></p>
<p><b>Quality:</b> How clear is the audio on average? Does it sound like it's coming from a walkie-talkie (bad quality) or a studio-quality sound system (excellent quality)?</p>
<p><b>Diversity:</b> How diverse are the speakers in the recordings on average? Do they mostly sound similar (not at all diverse) or are there many speakers represented (extremely diverse)?</p>

<div class="form-group">


<table>
	<tbody>
        <tr>
            <th class="noborder"></th>
            <th colspan=10><div class="thlab"><b>Digit Classification</b></div></th>
            <th class="noborder"></th>
            <th colspan=5><div class="thlab"><b>Digit Intelligibility</b></div></th>
        </tr>
		<tr>
			<th class="noborder"></th>
			<th><div class="thlab">0</div><div>Zero</div></th>
            <th><div class="thlab">1</div><div>One</div></th>
            <th><div class="thlab">2</div><div>Two</div></th>
            <th><div class="thlab">3</div><div>Three</div></th>
            <th><div class="thlab">4</div><div>Four</div></th>
            <th><div class="thlab">5</div><div>Five</div></th>
            <th><div class="thlab">6</div><div>Six</div></th>
            <th><div class="thlab">7</div><div>Seven</div></th>
            <th><div class="thlab">8</div><div>Eight</div></th>
            <th><div class="thlab">9</div><div>Nine</div></th>
            <th class="noborder"></th>
			<th><div class="thlab"><b>1: Not at all</b></div><div>Not at all intelligible</div></th>
			<th><div class="thlab"><b>2: Slightly</b></div><div>Slightly intelligible</div></th>
			<th><div class="thlab"><b>3: Moderately</b></div><div>Moderately intelligible</div></th>
			<th><div class="thlab"><b>4: Very</b></div><div>Very intelligible</div></th>
			<th><div class="thlab"><b>5: Extremely</b></div><div>Extremely intelligible</div></th>
		</tr>
                {rows}
	</tbody>
</table>


<table>
	<tbody>
        <tr>
            <th class="noborder"></th>
            <th colspan=5><div class="thlab"><b>Audio Quality</b></div></th>
            <th class="noborder"></th>
            <th colspan=5><div class="thlab"><b>Speaker Diversity</b></div></th>
        </tr>
		<tr>
			<th class="noborder"></th>
			<th><div class="thlab"><b>1: Bad</b></div><div>Very noisy audio</div></th>
			<th><div class="thlab"><b>2: Poor</b></div><div>Mostly noisy audio</div></th>
			<th><div class="thlab"><b>3: Fair</b></div><div>Somewhat clear audio</div></th>
			<th><div class="thlab"><b>4: Good</b></div><div>Mostly clear audio</div></th>
			<th><div class="thlab"><b>5: Excellent</b></div><div>Clear audio</div></th>
            <th class="noborder"></th>
			<th><div class="thlab"><b>1: Not at all</b></div><div>Not at all diverse (none or almost no distinct speakers)</div></th>
			<th><div class="thlab"><b>2: Slightly</b></div><div>Slightly diverse (few distinct speakers)</div></th>
			<th><div class="thlab"><b>3: Moderately</b></div><div>Moderately diverse (many distinct speakers) </div></th>
			<th><div class="thlab"><b>4: Very</b></div><div>Very diverse (almost all distinct speakers)</div></th>
			<th><div class="thlab"><b>5: Extremely</b></div><div>Extremely diverse (all distinct speakers)</div></th>
		</tr>
        <tr>
            <th class="noborder"></th>
            <td><input class="form-control" type="radio" required="" name="quality" value="1"></td>
			<td><input class="form-control" type="radio" required="" name="quality" value="2"></td>
			<td><input class="form-control" type="radio" required="" name="quality" value="3"></td>
			<td><input class="form-control" type="radio" required="" name="quality" value="4"></td>
			<td><input class="form-control" type="radio" required="" name="quality" value="5"></td>
            <th class="noborder"></th>
            <td><input class="form-control" type="radio" required="" name="diversity" value="1"></td>
			<td><input class="form-control" type="radio" required="" name="diversity" value="2"></td>
			<td><input class="form-control" type="radio" required="" name="diversity" value="3"></td>
			<td><input class="form-control" type="radio" required="" name="diversity" value="4"></td>
			<td><input class="form-control" type="radio" required="" name="diversity" value="5"></td>
        </tr>
	</tbody>
</table>

<input type="submit">

</div>

</html>
"""

ROW_TEMPL = """
		<tr>
			<td><audio controls=""><source src="${{recording_{i}_url}}" type="audio/mpeg"/></audio></td>
            <td><input class="form-control" type="radio" required="" name="recording_{i}_digit" value="0"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_digit" value="1"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_digit" value="2"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_digit" value="3"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_digit" value="4"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_digit" value="5"></td>
            <td><input class="form-control" type="radio" required="" name="recording_{i}_digit" value="6"></td>
            <td><input class="form-control" type="radio" required="" name="recording_{i}_digit" value="7"></td>
            <td><input class="form-control" type="radio" required="" name="recording_{i}_digit" value="8"></td>
            <td><input class="form-control" type="radio" required="" name="recording_{i}_digit" value="9"></td>
            <th class="noborder"></th>
            <td><input class="form-control" type="radio" required="" name="recording_{i}_intelligibility" value="1"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_intelligibility" value="2"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_intelligibility" value="3"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_intelligibility" value="4"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_intelligibility" value="5"></td>
		</tr>
"""

import sys

n = int(sys.argv[1])

rows = []
for i in range(n):
  rows.append(ROW_TEMPL.format(i=i))
rows = '\n'.join(rows)

print(TEMPL.format(rows=rows))
