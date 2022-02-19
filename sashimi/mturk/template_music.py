
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

<h2>Rate the audio fidelity and musicality of piano music.</h2>

<p><b>Please use headphones in a quiet environment if possible.</b></p>
<p><b>Some files may be loud, so we recommend keeping volumes at a moderate level.</b></p>
<p>You will be presented a batch of recordings and asked to rate each of them on audio fidelity and musicality.</p>
<p>Some are computer generated, while others are performed by a human.</p>
<p><b>Fidelity:</b> How clear is the audio? Does it sound like it's coming from a walkie-talkie (bad fidelity) or a studio-quality sound system (excellent fidelity)?</p>
<p><b>Musicality:</b> To what extent does the recording sound like real piano music? Does it change in unusual ways (bad musicality) or is it musically consistent (excellent musicality)?</p>

<div class="form-group">

<table>
	<tbody>
        <tr>
            <th class="noborder"></th>
            <th colspan=5><div class="thlab"><b>Fidelity</b></div></th>
            <th class="noborder"></th>
            <th colspan=5><div class="thlab"><b>Musicality</b></div></th>
        </tr>
		<tr>
			<th class="noborder"></th>
			<th><div class="thlab"><b>1: Bad</b></div><div>Very noisy audio</div></th>
			<th><div class="thlab"><b>2: Poor</b></div><div>Mostly noisy audio</div></th>
			<th><div class="thlab"><b>3: Fair</b></div><div>Somewhat clear audio</div></th>
			<th><div class="thlab"><b>4: Good</b></div><div>Mostly clear audio</div></th>
			<th><div class="thlab"><b>5: Excellent</b></div><div>Clear audio</div></th>
            <th class="noborder"></th>
            <th><div class="thlab"><b>1: Not at all</b></div><div>Not musical at all</div></th>
			<th><div class="thlab"><b>2: Slightly</b></div><div>Somewhat musical</div></th>
			<th><div class="thlab"><b>3: Moderately</b></div><div>Moderately musical</div></th>
			<th><div class="thlab"><b>4: Very</b></div><div>Very musical</div></th>
			<th><div class="thlab"><b>5: Extremely</b></div><div>Extremely musical</div></th>
		</tr>
                {rows}
	</tbody>
</table>

<input type="submit">

</div>

</html>
"""

ROW_TEMPL = """
		<tr>
			<td><audio controls=""><source src="${{recording_{i}_url}}" type="audio/mpeg"/></audio></td>

			<td><input class="form-control" type="radio" required="" name="recording_{i}_quality" value="1"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_quality" value="2"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_quality" value="3"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_quality" value="4"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_quality" value="5"></td>
            <th class="noborder"></th>
            <td><input class="form-control" type="radio" required="" name="recording_{i}_musicality" value="1"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_musicality" value="2"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_musicality" value="3"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_musicality" value="4"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_musicality" value="5"></td>
		</tr>
"""

import sys

n = int(sys.argv[1])

rows = []
for i in range(n):
  rows.append(ROW_TEMPL.format(i=i))
rows = '\n'.join(rows)

print(TEMPL.format(rows=rows))
