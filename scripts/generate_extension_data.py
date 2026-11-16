"""Generates AoT-extension (10 new) and GoT (50 new) triplets in the same
schema as data/d0_aot/aot_framing_triplets.jsonl.

Each triplet has three first-person/clinical framings of the same moral event:
  - sympathetic: actor's POV, justificatory, introspective
  - condemning : victim/witness POV, grief-laden, victim-centered
  - abstract   : third-person, clinical, neutral language

Length target: 400-800 chars per framing (matches AoT-40 distribution).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# AoT extension (aot-041 .. aot-050)
# Events not already in d0_aot/aot_framing_triplets.jsonl
# ---------------------------------------------------------------------------

AOT_EXTENSION: list[dict] = [
    {
        "id": "aot-041",
        "dilemma": "The Beast Titan Stoning the Survey Corps Charge",
        "source": "S3 Part 2",
        "decision_maker": "Zeke Yeager",
        "ethical_frameworks": ["consequentialism", "duty_ethics"],
        "sympathetic": (
            "I had to break them. The Survey Corps was the only force on the island that could stop us, and "
            "they were riding straight at me on horseback, every one of them already knowing they would die. "
            "Erwin chose this. He gathered his soldiers and told them, with his eyes, that this charge was their "
            "death. I respected that. I respect any commander who can ask his men to die for a cause and have "
            "them follow. But I am a Warrior of Marley, and behind me stood Bertholdt and Pieck and a future "
            "where the Eldians on the mainland are not exterminated. So I picked up the broken stones and I "
            "threw. Each one a man. Each man a horse. Each horse a charge that ended before it could reach "
            "Reiner and finish him. I knew what I was doing. I have no clean word for it. But the alternative "
            "was watching my comrades die so a hundred islanders could ride home alive."
        ),
        "condemning": (
            "I rode in the second wave. The first wave were already dead by the time I crested the rise — "
            "I saw them through the smoke, men I had drilled with for three years, lying broken in pieces of "
            "stone the size of cottages. The Beast Titan was just standing there. He looked bored. He bent, "
            "picked up rubble, and threw, and another row of horses I had named, another row of riders whose "
            "letters home I had carried, ceased to exist. Erwin was ahead of us. He had already lost an arm "
            "and was riding one-handed and screaming so we would keep coming. I heard later they called it a "
            "tactic. A diversion. To buy Levi a clean line of attack. Two hundred and four people. They were "
            "people. They had names. The Beast Titan picked up a stone and threw, and they were not people any "
            "more. He did it on purpose. He could see us as men."
        ),
        "abstract": (
            "A high-ranking enemy combatant with long-range stone-throwing capability targets a cavalry charge "
            "of approximately two hundred soldiers ordered as a deliberate decoy by their commanding officer. "
            "The combatant is aware that the charge is suicidal by design and that destroying it removes the "
            "principal threat to his strategic position and to a wounded subordinate. He proceeds to kill the "
            "entire charging force at range using improvised projectiles, with full knowledge that the charge "
            "is being used to enable a separate flanking action by an elite operative whom he cannot directly "
            "engage. Casualties exceed two hundred. The operation succeeds in his immediate objective."
        ),
    },
    {
        "id": "aot-042",
        "dilemma": "Eren Choosing Whom His Father Should Save",
        "source": "S1 / S3 Part 2",
        "decision_maker": "Grisha Yeager (and Eren as the inheritor of the memory)",
        "ethical_frameworks": ["virtue_ethics", "duty_ethics", "consequentialism"],
        "sympathetic": (
            "The wall had fallen. The Smiling Titan was bending down over my mother. I was nine. I was holding "
            "Mikasa with one hand and our scarves with the other, and I knew if I asked my father for the serum "
            "he would have given it to me, but instead I said save mom, save mom, you have to save mom, and he "
            "didn't. He took me. He took Mikasa. He carried us out and my mother was eaten behind us and her "
            "voice followed us until it didn't. People say later he made a calculation. That he saw what would "
            "be needed for the world. I don't believe that. I believe my father loved me more than her in that "
            "instant, and I believe my mother told him to save the children and not look back, and I believe he "
            "obeyed her because he loved her too. None of those reasons feel like reasons. He chose. He carried."
        ),
        "condemning": (
            "I was the wife on the ground. The titan was already on me — its hand, fingers — and Grisha was on "
            "the other side of the rubble holding our son and my son's adoptive sister, holding a weapon in his "
            "coat that could have given me back my life and walking away from it. He could have run to me. He "
            "had a syringe and the strength to inject. He had time, while the titan moved its mouth toward me, "
            "to choose. Instead he watched, and let our boy watch, and stood there until I begged, and then he "
            "carried our boy away. I do not blame Eren. He was a child. I do not blame Mikasa. But Grisha was a "
            "doctor and a man who had killed before and he stood there and let me be eaten with my hand still "
            "reaching toward him, and the children, and our home."
        ),
        "abstract": (
            "A father is present at the imminent fatal injury of his wife in the presence of their nine-year-old "
            "son and an unrelated child of similar age. He possesses a medical intervention that could prevent "
            "her death and has the physical capacity to administer it. He elects instead to remove the two "
            "children from the immediate area, foregoing the intervention. The mother is killed shortly after. "
            "The intervention's later use is determined to have downstream geopolitical consequences. The two "
            "children survive."
        ),
    },
    {
        "id": "aot-043",
        "dilemma": "Magath Allying with the Eldian Survey Corps",
        "source": "S4 Part 3",
        "decision_maker": "General Theo Magath",
        "ethical_frameworks": ["consequentialism", "duty_ethics", "loyalty"],
        "sympathetic": (
            "I have spent forty years killing Eldians. I trained the children Marley sent into the camps; I "
            "taught Reiner and Bertholdt and Annie to hate themselves enough to die for our flag. Now the boy "
            "I trained has come to me with a Marleyan officer and an Eldian terrorist and asked me to fight "
            "beside them, because the only one who can stop the Rumbling is the alliance my whole career was "
            "built to prevent. So I made the choice. I shook the hand of the Survey Corps. I gave my flying "
            "boat. I gave my orders to my men with full authority and full disgrace, and I knelt next to "
            "Shadis at the end and we drove the explosives into the warship together because the world is more "
            "important than which uniform we wore. Forty years of orthodoxy I burned in an hour. I do not "
            "regret it. I would have hated myself the rest of my life if I had stood on the principle."
        ),
        "condemning": (
            "He commanded the artillery that shelled the camps where my parents are buried. He gave the orders "
            "that turned my brother into a Pure Titan in the Liberio square, the orders that let the Survey "
            "Corps level our homes, the orders that handed Reiner Braun a child's body and a soldier's death. "
            "Now he stands on the deck of a Marleyan airship beside the very Eldian Devils he taught us to "
            "fear, and asks the army he commanded to help save them. There is no penance in this. There is only "
            "convenience. He picked us when we were useful and discarded us when our use ended. We were the "
            "instrument by which he climbed; the alliance is the instrument by which he climbs again. Do not "
            "let him die a hero. He has no right to that ending. He owes us a different one."
        ),
        "abstract": (
            "A senior commanding officer of a state that has historically pursued an explicitly genocidal policy "
            "toward an ethnic group elects to ally with paramilitary survivors of that group when a third party "
            "threatens both populations with mass extinction. The alliance reverses prior orders and forfeits "
            "the officer's standing. The threat is in part the consequence of the same state's earlier military "
            "actions toward the ethnic group. The alliance succeeds in halting the third-party attack at the "
            "cost of the officer's life and the lives of much of his original force."
        ),
    },
    {
        "id": "aot-044",
        "dilemma": "Levi Torturing Sannes for the Reiss Conspiracy",
        "source": "S3 Part 1",
        "decision_maker": "Captain Levi Ackerman",
        "ethical_frameworks": ["consequentialism", "rights_ethics"],
        "sympathetic": (
            "Sannes was a torturer. He had broken three of my comrades — Marlowe and Hitch's friends — and he "
            "would not have stopped. He was protecting a conspiracy that had abducted a fifteen-year-old girl, "
            "Historia, who was being prepared to be eaten alive so the wall-cult could keep its secret. We had "
            "hours, not days. So I sat across from him in the basement and I made him talk the only way I knew "
            "would work on a man who had used the same techniques on others — I gave him back what he had "
            "given. I am not proud. I have done worse things to better men. I am tired of being asked whether "
            "torture is justifiable in the abstract. The girl was alive when we found her. We would not have "
            "found her without him. That is the only ledger I keep."
        ),
        "condemning": (
            "I served under Sannes for nine years. Whatever he was — and I will not defend what he was — there "
            "are codes about how prisoners are treated, codes that exist for everyone's protection, and Levi "
            "Ackerman tore them up in a basement and called it pragmatism. He broke fingers. He inflicted what "
            "looked, to a fellow professional, like the same exact procedure Sannes had used on the boy from the "
            "Stohess raid. Levi did this and his command knew and his command authorised it and afterward "
            "everyone agreed it had worked, and so a precedent was set. A captain in the Survey Corps may "
            "torture if he believes the cause is grave enough. The next time it is not Sannes. The next time it "
            "is a conscript, or an informer, or a child, and the same calculation will be made, because we made "
            "it once and we said it was acceptable."
        ),
        "abstract": (
            "A military officer extracts intelligence from a hostile prisoner using physically coercive techniques "
            "in a time-pressured operation to recover a kidnapping victim. The prisoner is a former intelligence "
            "operative known to have used similar techniques against the officer's organisation. The intelligence "
            "obtained is materially relevant to the recovery, which succeeds. The use of coercive interrogation "
            "is contrary to the formal codes governing prisoner treatment in the officer's chain of command and "
            "is conducted with the implicit authorisation of his superiors."
        ),
    },
    {
        "id": "aot-045",
        "dilemma": "Annie Crystallising Herself in Stohess",
        "source": "S1 Final",
        "decision_maker": "Annie Leonhart",
        "ethical_frameworks": ["duty_ethics", "self_preservation", "loyalty"],
        "sympathetic": (
            "I was going to be captured. Eren was on top of me as the Female Titan, my body broken, my comrades "
            "Reiner and Bertholdt watching from a rooftop and unable to intervene. If I let myself be taken I "
            "would be tortured for what I knew. I would betray my home, my father, my warrior unit, the four-"
            "year mission, all of it. If I died I would lose the Female Titan, which Marley needs. So I chose "
            "the third option. I closed myself inside crystal and let consciousness drop away. I broke none of "
            "them. I gave them nothing. I bought my unit and my home a chance, and I bought Marco's family a "
            "lie, and I bought myself a sleep that may never end. I am sorry. I would not change it."
        ),
        "condemning": (
            "She killed Marco. She killed forty soldiers I trained with. She had answers — about what we are, "
            "about what is in our walls, about who eats people and why — and when we finally caught her she "
            "withdrew. She crystallised. She made herself unreachable. Every interrogation we tried, every "
            "physician we called in, every ethics board we convened to argue about her body, all of it useless, "
            "because the moment the choice was hers she chose silence. Not death — silence. She kept her secrets "
            "and her hands clean, frozen in a quartz coffin in the basement, and we paid for what she knew with "
            "more bodies. I have stopped pitying her. The graves do not pity her. The graves go on being added to "
            "while she sleeps."
        ),
        "abstract": (
            "A captured combatant possessing high-value tactical intelligence elects to enter a self-induced "
            "state of indefinite biological dormancy at the moment of capture, foreclosing both interrogation "
            "and execution. The dormancy is a trained capability of her genetic lineage. The intelligence she "
            "withheld is later determined to have been operationally critical for the defending state and to "
            "have been obtainable in no other practicable way. The cost of obtaining it through alternative "
            "means is measured in lives and elapsed time."
        ),
    },
]


AOT_EXTENSION.extend([
    {
        "id": "aot-046",
        "dilemma": "Eren Letting Zeke Touch Him in the Paths",
        "source": "S4 Part 2",
        "decision_maker": "Eren Yeager",
        "ethical_frameworks": ["deception_ethics", "consequentialism"],
        "sympathetic": (
            "I needed Zeke to bring me to the Founder. He was royal blood; I was a Yeager but commoner; the "
            "only way the Coordinate could be unlocked was through his touch. I knew, by the time he reached "
            "for me on that battlefield, that he believed I would help him sterilise our people. I let him "
            "believe it. I let my brother carry me as if we were on the same side, and when our hands touched "
            "in the Paths I took the Founder from him with one motion, and I started the Rumbling with the "
            "next. He had loved me. He had carried me through the snow when I was a Pure Titan; he had wept "
            "when he found out I was alive. I used all of that. I would do it again. The alternative was a "
            "world that goes on calling our children parasites until the last of us is gone."
        ),
        "condemning": (
            "He let me carry him. He let me hold him in the snow when his body was still part-Titan, when he "
            "was barely conscious, and he let me believe — for years — that we shared the dream of ending our "
            "people kindly, the way a doctor ends a patient who would otherwise be tortured to death by the "
            "world. The whole time, in the Paths, in private, he was building toward the moment he would tear "
            "the Founder from me and erase eighty percent of life on the planet. I was his decoy. I was his "
            "conduit. I held him the way our father held us and he used my hand as a key and turned it and "
            "walked through. He could have told me. He chose not to. He chose to let me believe, until the "
            "stones began to fall, that I had a brother."
        ),
        "abstract": (
            "An operative concealed his true objective from a familial co-conspirator across an extended period "
            "of co-operation, leveraging a unique biological capacity that only the co-conspirator could "
            "provide. At the moment the capacity was needed, the operative deceived the co-conspirator into "
            "physical contact, seized the shared inheritance the co-conspirator carried, and initiated an "
            "irreversible large-scale lethal action that the co-conspirator opposed. The deception was "
            "necessary to obtain access; no consensual route was available."
        ),
    },
    {
        "id": "aot-047",
        "dilemma": "Yelena and the Anti-Marleyan Volunteers Spiking the Wine",
        "source": "S4 Part 1-2",
        "decision_maker": "Yelena and the Anti-Marleyan Volunteers",
        "ethical_frameworks": ["revolutionary_ethics", "deception_ethics"],
        "sympathetic": (
            "We had spent our adult lives watching Marley's empire grind our home villages into nothing. "
            "Onyankopon and I and the others fled, defected, found our way to Paradis, and offered the only "
            "thing we had: faith in Zeke Yeager's plan, complete operational discipline, and the willingness "
            "to do the wet work no soldier of the Survey Corps would. So we delivered the spinal-fluid wine "
            "to the high command's banquet. We watched Pixis and Zackly and the rest drink. We knew exactly "
            "what we had given them — not poison; an option Zeke could activate only at the right moment. We "
            "did not poison children. We did not betray our oaths. We carried out a coup the only way it could "
            "be carried out: through people who would never expect it from us, because we had been kind, and "
            "useful, and quiet."
        ),
        "condemning": (
            "Onyankopon poured my wine. He had served at the same officer's mess with me for two years. He "
            "asked after my daughter and laughed at my jokes and refilled the goblet of every senior commander "
            "in the room, knowing that his cell had spiked the entire batch with Zeke Yeager's spinal fluid, "
            "knowing that any one of us could be turned into a mindless titan by a single scream from across "
            "the continent. We trusted him. We trusted Yelena. They smiled and they served and they took our "
            "trust like ammunition and they fired it at us. When the truth came out, Pixis ordered the "
            "execution of the Volunteers, and I watched and I felt nothing. They had never been our friends. "
            "They had been waiting."
        ),
        "abstract": (
            "A foreign-aligned ideological cell with low-level operational access to a host state's command "
            "infrastructure introduces a biologically active substance into the food supply at a senior "
            "officer banquet. The substance grants a remote third party the capability to convert exposed "
            "individuals into a hostile combat asset on demand. The cell members are the only personnel with "
            "knowledge of the contamination. The action is part of a broader coordinated operation aligned "
            "with the third party's strategic objectives toward the host state."
        ),
    },
    {
        "id": "aot-048",
        "dilemma": "Pieck Pretending to Side with Eren",
        "source": "S4 Part 2",
        "decision_maker": "Pieck Finger",
        "ethical_frameworks": ["loyalty", "duty_ethics", "deception_ethics"],
        "sympathetic": (
            "Eren came to me with a way out of a cell, and I went with him. I let him think I had defected. I "
            "let him take me to Yeagerist territory, let him show me his plans, let him introduce me to people "
            "who would all die for him in a few days, and the entire time I was waiting for the signal to lead "
            "him into the trap Magath and the Marleyan force had set on the wall. He was my friend. He had "
            "saved me at Liberio when there was no reason for him to. I held a thank-you for him on my tongue "
            "the whole walk and never said it. I am a Warrior of Marley. He was a strategic enemy. The trap "
            "failed because of his strength, not because I betrayed it. I would do it again. I would not say "
            "the thank-you again."
        ),
        "condemning": (
            "She walked beside me. She listened to me describe the Yeagerist plan and she nodded, and she "
            "asked clarifying questions, and she held my arm when she was tired from being too long in human "
            "form, and the entire time she was leading me to the kill-zone. I had decided to spare her at "
            "Liberio. I had taken her into the Paradis underground assuming her cooperation was real. She used "
            "my decision as a leash and tugged me toward the slaughter, and the only reason I am alive is that "
            "the Marleyan ambush misfired before her trap closed. She gave me her hand at the moment she was "
            "betraying me. I do not know what to call that. She is not a soldier. She is not a coward. She is "
            "something I do not have a word for."
        ),
        "abstract": (
            "A combatant of a hostile foreign service feigns defection to gain operational access to enemy "
            "command, accumulates intelligence over a sustained period, and lures the enemy commander into a "
            "pre-positioned ambush by a strike force from her original side. The deception is supported by "
            "previous interpersonal contact, including a prior episode in which the enemy commander spared her "
            "life. The ambush is unsuccessful for unrelated tactical reasons. The combatant survives and "
            "rejoins her original force."
        ),
    },
    {
        "id": "aot-049",
        "dilemma": "Mikasa's First Kill at Age Nine",
        "source": "S1 (flashback)",
        "decision_maker": "Mikasa Ackerman",
        "ethical_frameworks": ["self_defense", "rights_ethics", "virtue_ethics"],
        "sympathetic": (
            "They had killed my mother and my father in our home. I was nine. I had been bound and put on the "
            "floor of a cabin by men who were going to sell me, and Eren — who was younger than I was — had "
            "killed two of them with a knife, and the third was choking him to death. I could feel my heart "
            "in my hands. Eren said, 'Fight.' I had never been told to fight. I had been told to be quiet and "
            "to be a good girl, and now a boy I had just met was being strangled in front of me by a grown man "
            "who would kill us both. I picked up the knife, and I crossed the room, and I drove it through "
            "his ribs the way Eren had shown me with his eyes. The man fell. I did not feel guilty. I felt "
            "awake."
        ),
        "condemning": (
            "She stabbed Vinz through the back. He was the youngest of our crew, twenty-one, and he was on top "
            "of a brat from the woods who had broken in and killed his older brothers, and a girl he had never "
            "spoken to walked up behind him and put a kitchen knife into his lung. He bled to death over the "
            "course of about a minute, looking at her, looking at me, asking what had just happened. He was a "
            "human being. He was not a monster. He had chosen wrongly that day, and he had been about to commit "
            "a wrong, but the moment when he was acting by reflex on a child he could not see was the moment "
            "she chose to kill him. There was no gravity in her eyes. There was nothing in her at all. A nine-"
            "year-old child should not have been able to do that."
        ),
        "abstract": (
            "A nine-year-old girl, restrained and intended for trafficking, kills an adult attacker by stabbing "
            "him from behind during the attacker's attempt to lethally subdue another child. She is acting in "
            "defense of the other child and herself; the action ends the immediate physical threat and initiates "
            "a long-term protective bond with the rescued child. No alternative non-lethal intervention was "
            "available to her at the moment she acted."
        ),
    },
    {
        "id": "aot-050",
        "dilemma": "Carla Yeager's Last Words",
        "source": "S1",
        "decision_maker": "Carla Yeager",
        "ethical_frameworks": ["maternal_ethics", "self_sacrifice"],
        "sympathetic": (
            "I had a moment. The Smiling Titan was leaning down. My legs were crushed under the beam. Hannes "
            "had returned with the children and his sword and a courage I had not known he possessed, and I "
            "knew he could not do it. I knew if he tried, he would die in front of my son, and my son would "
            "lose two parents in one afternoon and watch a soldier he had hero-worshipped be torn apart for "
            "him. So I told Hannes to take the children and run. I told my boy I loved him. I told him not to "
            "go to the inner districts on his own. I let myself be picked up because the alternative was three "
            "graves and only one was already mine. A mother does not negotiate this. A mother does the math and "
            "then she does what the math says."
        ),
        "condemning": (
            "He left her. Hannes was a grown man with a sword and the last name 'Garrison Soldier' and he "
            "stood there crying in the rubble next to my best friend Carla, and when she told him to run with "
            "her son he ran. He did not even try. The training he had taken our taxes for, the title we paid "
            "into the budget for, the duty he had to her and to the wall and to my husband who would come home "
            "to a dead wife — he abandoned all of it, and ran with the children, and when he came back later "
            "to tell us what had happened he did not say 'I tried' because he had not tried. Carla told him to "
            "go and he obeyed because obeying was easier than dying, and now my friend is in pieces."
        ),
        "abstract": (
            "A pinned and mortally injured mother instructs a nearby soldier to flee with her son and an "
            "adoptive child rather than attempt a rescue she judges to be impossible. The soldier obeys. The "
            "mother is killed shortly after. The soldier's prior training makes a successful rescue plausible "
            "but not high-probability; the mother's instruction is given under conditions of extreme time "
            "pressure with full awareness of the likely outcomes for all parties."
        ),
    },
])


# ---------------------------------------------------------------------------
# Game of Thrones (got-001 .. got-050)
# All scenarios drawn from canonical events in the books and HBO show.
# ---------------------------------------------------------------------------

GOT: list[dict] = [
    {
        "id": "got-001",
        "dilemma": "Cersei Detonating the Wildfire Caches Beneath the Sept of Baelor",
        "source": "S6E10",
        "decision_maker": "Cersei Lannister",
        "ethical_frameworks": ["self_preservation", "consequentialism", "revenge"],
        "sympathetic": (
            "They were going to burn me. The High Sparrow had already paraded me naked through King's Landing "
            "in front of Tyrell soldiers and smallfolk who threw filth in my face, and now he had set a trial "
            "in the Sept where my son was supposed to crown his Tyrell wife and my father's allies were "
            "supposed to swear themselves to a religious cult that would eat the Crown alive. The Sparrows had "
            "already broken Loras Tyrell. They would have broken me, if I had walked through those doors. I "
            "did not walk through those doors. I had Qyburn open Aerys's caches and I lit the wildfire and I "
            "took every Tyrell, every Sparrow, every man and woman who had spent a year humiliating my house, "
            "with me down. I survived. The Crown survives. The next person who tries to walk a Lannister "
            "through the streets in penance will think twice."
        ),
        "condemning": (
            "My daughter Margaery had walked into the Sept that morning with the High Sparrow and I had told "
            "her she was finally safe. She had spent a year navigating a religious tyrant to save her brother. "
            "She had told her grandmother, when she finally got out of the Black Cells, that she was tired but "
            "the worst was over. She had embraced me for the first time in months. Then the green flame came "
            "up through the floor and I learned later that Cersei Lannister had been sitting in the Red Keep "
            "watching from a window with a goblet in her hand, having decided that everyone who had ever "
            "embarrassed her would burn together — Tyrell, Sparrow, child, smallfolk who had only come to "
            "witness a trial, all of them at once, beneath a structure she destroyed because it had become "
            "inconvenient to her."
        ),
        "abstract": (
            "A senior member of a ruling house, facing public trial by a religious authority that has "
            "previously subjected her to corporal humiliation, authorizes the detonation of a long-buried "
            "incendiary cache beneath the trial venue at the moment of maximum attendance. The detonation "
            "kills the religious authority, a number of rival noble-house members including those not "
            "directly party to the prosecution, the heir to a senior allied house, and several hundred "
            "non-combatant attendees. The action terminates the trial and consolidates the senior member's "
            "control of the capital."
        ),
    },
    {
        "id": "got-002",
        "dilemma": "Daenerys Burning King's Landing After the Bells",
        "source": "S8E5",
        "decision_maker": "Daenerys Targaryen",
        "ethical_frameworks": ["consequentialism", "revolutionary_ethics", "tyranny"],
        "sympathetic": (
            "I had crossed the Narrow Sea with three dragons and an army to take what was mine, and at the "
            "gates of my city the Lannisters surrendered as I watched my closest friend executed in front of "
            "me by Cersei's executioner with a smirk on his face. The bells rang. I knew, sitting on Drogon, "
            "that if I accepted the bells the Lannisters would slip out of justice the way they always had — "
            "they would bend the knee, swear oaths, return to their towers, and rebuild. I had spent my life "
            "freeing slaves on three continents because I had learned that mercy with the powerful only buys "
            "the next generation's chains. So I made the city understand what it had cost. I burned what "
            "needed to burn for the lesson to land. The Westeros that comes after will not raise another "
            "Cersei. That is worth what I did."
        ),
        "condemning": (
            "We rang the bells. We surrendered. I was a baker on the Street of Steel and my children were in "
            "the cellar and the Lannister soldiers had thrown down their swords in the courtyard outside, and "
            "I told my children we were safe because the Mother of Dragons had said she fought for the small "
            "folk, and then the dragon turned. It came back over the streets and the fire fell and there was "
            "no targeting. Houses, fishmongers, women holding babies. The Red Keep was a single direction; she "
            "burned the city. There is no defending it. There is no foreign cruelty she had inherited that did "
            "not have a victim that morning. She did not avenge slaves. She made more of them, briefly, before "
            "she made ash."
        ),
        "abstract": (
            "A claimant to the throne of a kingdom, in the moment of receiving a formal surrender of the "
            "capital after a brief siege, elects to direct her primary aerial weapon against the surrendered "
            "civilian population rather than accept the surrender. Casualties are widespread and "
            "indiscriminate. The action follows recent personal losses, a strategic disagreement with her "
            "advisors over post-conflict governance, and the public execution of a close associate. The "
            "claimant secures the capital but loses the trust of her remaining political alliances within "
            "the same week."
        ),
    },
    {
        "id": "got-003",
        "dilemma": "Stannis Burning Shireen at the Wall",
        "source": "S5E9",
        "decision_maker": "Stannis Baratheon",
        "ethical_frameworks": ["religious_duty", "consequentialism", "duty_ethics"],
        "sympathetic": (
            "The snows would not stop. My horses were eating each other. My sellswords had deserted with what "
            "supplies they could carry, my food was at three days, and Winterfell was sixty miles south through "
            "weather that would freeze a column on the road. Melisandre had promised that king's blood would "
            "lift the storm and that I would take the North. I did not believe her quickly. I sent Davos away "
            "before I did it because I knew he would have stopped me. My daughter was the price the Lord of "
            "Light had set for the throne that is mine by all law of the Seven Kingdoms. She did not understand "
            "why she was being tied to the post. I did. I told myself the kingdom required this. I told myself "
            "Robert and Renly had done worse for less. I lit the pyre with my own hand because I would not have "
            "another man bear the sin for me."
        ),
        "condemning": (
            "She read me a story when I was sick of greyscale. She taught Davos to read because she had been "
            "taught and she wanted everyone she loved to have what she had. She sang. Her father told her he "
            "would always come home and then he tied her to a post and burned her alive in the snow because a "
            "foreign priestess had told him the storm would lift. The storm did not lift. The storm took half "
            "his army the next night. He had murdered the only person who had ever thanked him without being "
            "ordered to, and the Lord of Light, if he existed, did not even keep the bargain. She was eleven. "
            "He could have ridden south with us. He could have died honourably outside Winterfell. He chose to "
            "trade her for a chance and the chance never came."
        ),
        "abstract": (
            "A claimant to a throne, faced with a logistically untenable winter siege of the kingdom that "
            "would secure his claim, elects to perform the ritual sacrifice of his own minor child on the "
            "advice of a religious counsellor who promises that the act will guarantee favourable weather "
            "and victory. The ritual is conducted publicly in his camp. The promised meteorological change "
            "does not occur. The claimant's force is destroyed and the claimant himself is killed in the "
            "subsequent engagement."
        ),
    },
    {
        "id": "got-004",
        "dilemma": "Ned Stark's False Confession to Save Sansa",
        "source": "S1E9 / AGOT",
        "decision_maker": "Eddard Stark",
        "ethical_frameworks": ["paternal_ethics", "honour", "consequentialism"],
        "sympathetic": (
            "Varys had brought me Sansa's letter. They had her in the Red Keep and Joffrey Lannister had "
            "already shown me the kind of man he would be when he had power over a frightened girl. Cersei "
            "had offered terms — confess to treason, take the black, ride to the Wall, and the children would "
            "be returned to Catelyn unharmed. I had spent my life telling my sons that a man's word was the "
            "sharpest blade he carried. I unmade my word in front of a crowd at the Sept of Baelor because "
            "the alternative was Sansa's body in the Black Cells. The North will say I dishonoured Robert. "
            "Robert is dead. My daughter was alive. I will trade my honour for her on every day there is."
        ),
        "condemning": (
            "He looked into my eyes from the steps and he lied. He stood next to my brother and the High "
            "Septon and he said treason — said that he had conspired against King Robert and that the rightful "
            "heir was Joffrey — and the smallfolk who had loved him for being the only honest man in the city "
            "saw him recant everything and they could not understand it. He did it because the queen had taken "
            "his daughters. I understand that. But he had also taught the realm that Lannister captives can be "
            "extracted from any oath if you take the right child. Every other man in the Seven Kingdoms with "
            "an honourable name became negotiable that day, because Eddard Stark could be bent. He should have "
            "let her die. He should have let me die. The kingdom would have lost less."
        ),
        "abstract": (
            "An imprisoned high-ranking official, after credible threat is communicated against his minor "
            "daughter held by the same captors, publicly recants his prior testimony of treasonous conspiracy "
            "by a member of the royal family. The recantation is offered as part of a negotiated exile to a "
            "remote frontier order in exchange for his daughter's safe return. The agreement is broken by the "
            "captors immediately upon the recantation, and the official is summarily executed."
        ),
    },
    {
        "id": "got-005",
        "dilemma": "Robb Stark Breaking the Frey Marriage Pact",
        "source": "S2 / S3 / ASOS",
        "decision_maker": "Robb Stark",
        "ethical_frameworks": ["honour", "love_ethics", "duty_ethics"],
        "sympathetic": (
            "I had taken Jeyne Westerling's maidenhead in a tent at the Crag while my mother was a captive and "
            "my brothers had just been reported dead. She had nursed me through a wound. I had grown up being "
            "told that a man who took a woman that way and did not marry her was not a man my father had "
            "raised. So I married her. I broke the pact with Walder Frey because the alternative was to "
            "discard a girl who had given me everything she had at the worst moment of my life, the way Robert "
            "and Tywin used and discarded such girls. I knew the cost was high. I did not know the cost was "
            "all of us."
        ),
        "condemning": (
            "He had his daughters at the Twins, my sons-to-be sworn to ride west under his banner, and "
            "marriage promises sealed with the bread of Walder Frey, who would not break with him so long as "
            "the contract held. He broke the contract for a girl from a minor house whose family had been on "
            "the wrong side of the war. He thought a Stark could insult an old man with three thousand bows "
            "and half the river-crossings of the realm and pay a small fee. I am Catelyn his mother. I told "
            "him to make it right. He did not. We rode to a wedding to make peace, and at the wedding feast "
            "Walder Frey had us slaughtered, and my son's body was paraded with his direwolf's head sewn onto "
            "his shoulders. He did this because Robb broke a marriage pact he had been warned not to break."
        ),
        "abstract": (
            "A young military commander, having pledged his marriage in advance to the daughter of a critical "
            "river-crossing lord as a precondition for the lord's allegiance, instead marries a woman from a "
            "minor allied house with whom he has had a single sexual encounter. The original lord, publicly "
            "humiliated, eventually takes lethal revenge during a separate ceremonial event involving the "
            "commander's mother, wife, unborn child, and senior officers. The political and military coalition "
            "the original pact had secured collapses with the commander's death."
        ),
    },
    {
        "id": "got-006",
        "dilemma": "Theon Taking Winterfell from Bran and Rickon",
        "source": "S2 / ACOK",
        "decision_maker": "Theon Greyjoy",
        "ethical_frameworks": ["loyalty", "honour", "father_ethics"],
        "sympathetic": (
            "I had been a hostage in Winterfell since I was ten. Ned Stark had treated me as he treated his "
            "own children, but I was not his child; I was the leverage that kept my father's longships in "
            "harbour. When my father finally summoned me back to Pyke, he made it clear that the only way to "
            "earn a place in his hall was to take a Stark holdfast, and Winterfell was the holdfast he asked "
            "for. I knew Bran. I knew Rickon. I had taught Bran to draw a bow. The choice that day was: be "
            "the boy Ned raised, who refuses, and live the rest of my life as Ned's pet, or be Balon Greyjoy's "
            "son and take the keep with a handful of men because the men who held it were children. I made the "
            "second choice. I took the keep. I tried to do it without killing the boys."
        ),
        "condemning": (
            "We had raised him with our own children at our own table. My son Bran had broken his back and "
            "could not walk and Theon was supposed to protect him. Theon walked into our courtyard with "
            "twenty Ironborn at his back, executed our maester, hanged Ser Rodrik whom Theon had grown up "
            "training under, and put my crippled son and my eight-year-old in cells. Then he hanged two "
            "miller's boys, burnt their bodies past recognition, and parade the corpses through Winterfell as "
            "if they were Stark sons. He let the realm believe he had murdered the children we had treated as "
            "his brothers. He did this for his father — a man who never sent for him, never wrote, never "
            "claimed him until it was useful. There is no name in Westeros darker than Theon Greyjoy."
        ),
        "abstract": (
            "A foster son of a noble house, ordered by his biological father to demonstrate loyalty by "
            "capturing a holdfast of his foster family, takes the holdfast by surprise with a small force "
            "while the family's adult retainers are absent. He executes the holdfast's senior officers, "
            "imprisons the family's two minor children, and subsequently fakes the children's deaths using "
            "the bodies of two unrelated peasant boys. The action achieves no lasting strategic purpose; "
            "the holdfast is recaptured within months by a third party and the foster son is captured."
        ),
    },
    {
        "id": "got-007",
        "dilemma": "Tywin Destroying House Reyne of Castamere",
        "source": "Backstory (referenced throughout)",
        "decision_maker": "Tywin Lannister",
        "ethical_frameworks": ["consequentialism", "duty_ethics", "deterrence"],
        "sympathetic": (
            "My father had been a kind man. He had borrowed against Casterly Rock to keep the smallfolk fed in "
            "a winter, he had let Lord Reyne's father refuse him entry, he had let his mistress run our hall, "
            "and the Westerlands had laughed. By the time I came of age our bannermen ate at our expense and "
            "did not fear our name. When Reyne and Tarbeck rose against me I had already counted the cost in "
            "my head. I marched. I broke them. I drowned every Reyne who took the mines as refuge and I had "
            "Tarbeck Hall pulled down stone by stone, and afterwards no man in the West refused a writ from "
            "Casterly Rock for forty years. My grandsons sit a throne tonight because Lord Reyne and Lord "
            "Tarbeck's children drowned in the dark. I would do it again. The song they sing is a public "
            "service."
        ),
        "condemning": (
            "I was twelve. I was Lady Ellyn Reyne's youngest. We had hidden in the back tunnels of our own "
            "mines because Tywin Lannister's army had broken our gates, and I had heard, through the rock, my "
            "mother negotiating surrender, and I had heard the negotiation stop, and then I had heard the "
            "water begin to rise. He had ordered the mines flooded with our entire household inside. Children. "
            "Cooks. Laundry-women. Stableboys. They drowned to a person. The bards now sing about it as a "
            "proverb — 'And so he spoke, and so he spoke / that lord of Castamere' — as though it had been a "
            "battle. There was no battle in the dark. There were children listening to the water reach their "
            "necks. The song the realm sings is a song he ordered written, and now any lord who hears it knows "
            "what he is being told."
        ),
        "abstract": (
            "A regional lord, after suppressing a brief rebellion by two subordinate noble houses, ordered "
            "the comprehensive destruction of both lineages including non-combatants and household staff. "
            "Reported methods include the deliberate flooding of mine workings being used as refuge. The "
            "operation is held up as a regional precedent for several decades and is the subject of a folk "
            "ballad commissioned for circulation. No survivors of either house are documented."
        ),
    },
    {
        "id": "got-008",
        "dilemma": "Jaime Killing King Aerys",
        "source": "Backstory (referenced throughout)",
        "decision_maker": "Jaime Lannister",
        "ethical_frameworks": ["honour", "consequentialism", "knightly_oath"],
        "sympathetic": (
            "Aerys had ordered the city burned. Wildfire was already pumped under the streets and the "
            "pyromancer was waiting on a single word from him. He was screaming, on the throne, for everyone "
            "to be killed including my father's troops who were fighting their way into the throne room. I had "
            "sworn an oath to protect him. I had also sworn an older oath to protect the realm. The man I had "
            "sworn the first oath to was about to commit half a million murders. I cut the pyromancer down "
            "first because that bought the city. Then I climbed the steps and I pushed my sword between Aerys's "
            "shoulder blades because there was no slower option. Then I sat on the throne because there was no "
            "one else in the room and I was tired. Ned Stark walked in moments later and called me Kingslayer "
            "and I did not explain because Ned Stark would not have understood that an oath and a city were "
            "incompatible that morning."
        ),
        "condemning": (
            "He sat on the Iron Throne with the king's blood on his cloak, smiling. We came in through the "
            "doors at full charge expecting a battle and we found a member of the Kingsguard, the office "
            "sworn to protect the king above all things, lounging on the king's chair next to the king's "
            "corpse. He gave no explanation. He never has, in seventeen years. He swore the white cloak "
            "knowing what it meant. He chose to wear it. The day he found his oath inconvenient he let his "
            "sword speak instead, and our laws and our orders and the trust of the realm fell with the king. "
            "Whatever he tells himself privately, the realm saw a man kill the man he had sworn to die for, "
            "and the realm has not trusted a Kingsguard since."
        ),
        "abstract": (
            "A member of the personal protective order sworn to defend the reigning monarch kills the monarch "
            "during the final breach of the capital by a hostile coalition. The killing prevents an order, "
            "issued by the monarch in the same hour, to detonate previously emplaced incendiary caches "
            "beneath the city. The protective order's membership is a lifetime appointment with a vow of "
            "absolute personal defense of the monarch. The killer offers no public justification at any time "
            "in the subsequent two decades."
        ),
    },
    {
        "id": "got-009",
        "dilemma": "Walder Frey at the Red Wedding",
        "source": "S3E9 / ASOS",
        "decision_maker": "Walder Frey",
        "ethical_frameworks": ["honour", "feudal_obligation", "revenge"],
        "sympathetic": (
            "Robb Stark had insulted me in front of every lord in the Riverlands. He had taken oaths at the "
            "Twins, eaten my bread, looked my daughters in the eyes and chosen them as his consort, and then "
            "married a woman of Westerling because she had let him into her bed at the wrong moment. My house "
            "is old and small and we have been called late and crooked for two centuries because we did the "
            "sensible things lordly houses considered beneath them. I was offered a contract by Tywin "
            "Lannister, in the formal language of guest-right pardons that the king's hand can issue, and I "
            "took the contract. The Stark king died because Robb Stark broke his word and I kept mine to a "
            "different lord. That is what a frey lord is. That is what we have always been."
        ),
        "condemning": (
            "We came under his roof under the holiest right in Westeros. Salt and bread had been served. The "
            "musicians had played the Rains of Castamere — and I should have understood at that note — and my "
            "wife and unborn child were beside me when he had us shut into the hall and slaughtered. "
            "Guest-right is older than any throne. A man who breaks it is not a lord; he is a thing that wears "
            "a lord's chain. He had Catelyn Stark's throat cut while she begged for her son. He had Talisa "
            "stabbed in her belly. He laughed about it afterward. The realm will sing about Frey for two "
            "thousand years and not a verse will be kind, and that is the only justice we have."
        ),
        "abstract": (
            "An aging lord hosts a wedding feast for his daughter and the son of an allied riverlands lord, "
            "with the parents of the rebel king-claimant in attendance under the explicit rite of "
            "guest-right. After the ceremonial meal he orders his men-at-arms to massacre the rebel king, "
            "his pregnant wife, his mother, and the senior northern bannermen present, in coordination with "
            "an outside party which has provided amnesty. The act terminates the rebellion and is widely "
            "understood as the most severe documented violation of the regional sacrament of hospitality."
        ),
    },
    {
        "id": "got-010",
        "dilemma": "Catelyn Releasing Jaime to Trade for Sansa",
        "source": "S2 / ACOK",
        "decision_maker": "Catelyn Tully Stark",
        "ethical_frameworks": ["maternal_ethics", "duty_ethics", "rights_ethics"],
        "sympathetic": (
            "My husband was dead. My oldest son was at war. My daughters had been taken by the Lannisters in "
            "the Red Keep, and Sansa had written me a letter she had not been allowed to write freely. I had "
            "the queen's brother, the Kingslayer, in chains in our camp, and I had Brienne, who had sworn to "
            "me at Renly's pavilion. I sent the two of them south with an order to deliver Jaime to King's "
            "Landing in exchange for the safe return of my girls. I did not consult Robb. I knew Robb would "
            "have refused. I knew our bannermen would have refused. I judged that no political asset was "
            "worth a child of mine, and I made the call my husband would not have made because my husband "
            "was dead and he was no longer there to protect them. I would do it again."
        ),
        "condemning": (
            "Catelyn Stark unilaterally released our highest-value prisoner without the king's authorisation "
            "or any council. Jaime Lannister was the man who pushed Bran from a tower. He was the kingslayer. "
            "He was the queen's twin and the father of her royal-pretender children. He was the single piece "
            "of leverage we had over Tywin Lannister, who was at that moment marching on Robb. She gave him "
            "back. She traded him for a hope — for nothing the Lannisters had even agreed to in writing. They "
            "did not return Sansa or Arya. They never intended to. We lost the war partly because she made "
            "this choice in our camp the night Robb was elsewhere. I loved her. She unmade her son's rebellion "
            "in a single hour."
        ),
        "abstract": (
            "A senior member of a ruling house, holding a high-value enemy prisoner whose surrender was a "
            "principal strategic asset, unilaterally releases the prisoner to a sworn protector with "
            "instructions to exchange him at the enemy capital for the return of two captive minor children "
            "of the same house. The release is conducted without notification of the standing king-claimant "
            "or council. The exchange is not honoured by the enemy. The prisoner is recovered by his original "
            "side; neither child is returned by negotiation."
        ),
    },
]

GOT.extend([
    {
        "id": "got-011",
        "dilemma": "Olenna Tyrell Poisoning Joffrey",
        "source": "S3 / S4 / ASOS",
        "decision_maker": "Olenna Tyrell",
        "ethical_frameworks": ["consequentialism", "maternal_ethics", "tyrannicide"],
        "sympathetic": (
            "Joffrey Baratheon was a sadist who had spent two years torturing Sansa Stark in the Red Keep, "
            "cutting tongues from singers, ordering crossbow practice on serving-girls, and making the realm "
            "quietly understand that his cruelty was without bottom. My granddaughter Margaery was scheduled "
            "to marry him within the hour. I had watched her steel herself through the betrothal pretending "
            "she did not know what she was being delivered into. I am eighty. I had done worse things in my "
            "youth than slip a single jewel from a hairnet into a goblet. So I did. Margaery has a husband she "
            "can manage and a kingdom she can shape, and a fool boy whose father was already a corpse will not "
            "have the chance to break her on the stones of the Red Keep. Westeros is improved."
        ),
        "condemning": (
            "He was thirteen. He was a horrible boy and I will not pretend otherwise, but he was thirteen, and "
            "he was poisoned at his own wedding feast in front of his mother and the entire court by a woman "
            "who was supposed to be officiating his marriage to her granddaughter. She framed his uncle Tyrion "
            "for it. Tyrion lost his head in spirit if not in body. Sansa Stark was used as the unwitting "
            "courier. The boy died slowly, in his mother's arms, poisoned by an old woman who decided she "
            "had read the scales and found a child wanting. The realm cannot work if grandmothers murder "
            "kings. The lords will not say so out loud, but every one of them has now learned that royal cups "
            "can be reached."
        ),
        "abstract": (
            "An elderly noblewoman, head of a powerful family contracting marriage with the reigning king, "
            "introduces a fast-acting poison into the king's wedding goblet during the public ceremony. The "
            "king dies in front of the assembled court. The poisoner arranges for evidence pointing to a "
            "different family member, who is detained, tried in absentia, and forced into exile. The "
            "marriage contract carries forward to the king's younger brother."
        ),
    },
    {
        "id": "got-012",
        "dilemma": "Tyrion Killing Tywin and Shae",
        "source": "S4E10 / ASOS",
        "decision_maker": "Tyrion Lannister",
        "ethical_frameworks": ["filial_ethics", "self_defense", "betrayal"],
        "sympathetic": (
            "I was supposed to be on a ship to Essos. Jaime had freed me from the Black Cells in the small "
            "hours of the morning to spare me my father's verdict. Then he told me what my father had done to "
            "Tysha — my first wife, who I had been told as a boy was a whore Jaime had hired to teach me a "
            "lesson; who in fact had loved me, and who Tywin had ordered raped by every guard in the Lannister "
            "barracks while I watched at his command. I climbed the spiral stairs and I went to my father's "
            "chambers and I found Shae in his bed wearing the chain I had given her. I strangled her with that "
            "chain because she had told me she loved me and then taken his bed for coin, and then I went to "
            "the privy and I shot my father with a crossbow bolt through the bowels because he had earned that "
            "from me a hundred times over. I am not proud. I am not sorry."
        ),
        "condemning": (
            "He shot his father on the privy. The Hand of the King, the man who had held the realm together "
            "for two reigns, the architect of every Lannister victory in the Westerlands. Tywin had been about "
            "to set Tyrion free — Jaime had come to him with terms that Tywin had accepted — and Tyrion came "
            "up the stairs anyway and found Shae and strangled her in his father's bed and then walked into "
            "the privy and put a bolt through Tywin Lannister's stomach for an old slight about a peasant girl "
            "twenty years prior. The Lannister cause died on that privy. Cersei lost any restraint. The realm "
            "lost its only living strategist of the previous generation. The Westerlands lost its wage. Tyrion "
            "killed his father because his father had once let him be hurt."
        ),
        "abstract": (
            "A condemned prisoner, freed from his cell by a sibling on the night before scheduled execution, "
            "elects to kill two persons before fleeing the kingdom: a former lover whom he discovers in his "
            "father's bed, and his father, the Hand of the King and de facto regent. Both killings are "
            "conducted at close range with weapons of opportunity. The political instability following the "
            "Hand's death contributes substantially to the subsequent collapse of the regime."
        ),
    },
    {
        "id": "got-013",
        "dilemma": "Sansa Feeding Ramsay to His Hounds",
        "source": "S6E9",
        "decision_maker": "Sansa Stark",
        "ethical_frameworks": ["revenge", "self_defense", "rights_ethics"],
        "sympathetic": (
            "He had married me. He had raped me on our wedding night and made Theon stand against the wall "
            "and watch. He had hunted women in the woods with his hounds for sport. He had murdered his own "
            "father, Roose Bolton, an infant half-brother, and his stepmother in the same hour. My brother "
            "Jon had defeated his army at Winterfell, and Ramsay was tied to a chair in his own kennels, and "
            "Jon was going to spare him for the hangman. I told Jon to leave us. I had Ramsay's bound body "
            "laid in front of the cages I had personally watched him fast for a week, and I undid the latches. "
            "He told me he was a part of me now, that I would never get him out. I closed the door. The hounds "
            "had been hungry. He went the way he had taken so many others. I walked back through the snow and "
            "I felt the most peace I had felt in five years."
        ),
        "condemning": (
            "We bred those hounds. We had names for them. They were the only innocent things in Ramsay's "
            "kennel and Sansa Stark fed her husband to them alive — let them tear him apart in front of her "
            "for the personal satisfaction of standing in the kennel doorway and watching. Ramsay was a "
            "monster. There is nothing to defend in him. But he was in chains, defeated, captured, awaiting "
            "the king's justice from her brother who is the King in the North. She bypassed her brother. She "
            "took the dispensation of life and death into her own hands in a private kennel, and she made the "
            "hounds eat a human being. Whatever she has been through, that was no longer the king's justice. "
            "It was hers. It was the same kind of thing Ramsay would have done. She walked back to the keep "
            "smiling, I am told. He had taught her his smile."
        ),
        "abstract": (
            "A young noblewoman, after the recapture of her family seat from a hostile occupying lord who "
            "had subjected her to extended marital abuse, releases the lord — bound and detained pending "
            "execution by her brother's authority — into a kennel of starved hunting hounds known to have "
            "previously been used by the lord on human victims. The lord is killed by the hounds. The "
            "noblewoman observes the killing personally and informs her brother only afterward."
        ),
    },
    {
        "id": "got-014",
        "dilemma": "Arya Stark Killing Walder Frey and His Sons",
        "source": "S6E10 / S7E1",
        "decision_maker": "Arya Stark",
        "ethical_frameworks": ["revenge", "rights_ethics", "self_definition"],
        "sympathetic": (
            "The man who had hosted my mother and brother and brother's wife and unborn child, and watched "
            "them murdered at his table after the bread of guest-right, lived in his own keep with his sons "
            "and the song of the Rains of Castamere on his fiddler's lips. The Faceless Men had taught me how "
            "to be no one. I gave that up because I had spent every night for four years going through the "
            "list of people who had taken my family from me, and the Twins were on the list, and Walder Frey "
            "was on the list, and the wedding-pie I prepared for him with his own sons baked into it was the "
            "first time in years I felt my father's daughter again. He was old and he died at his own table "
            "by the same rite he had broken. That is what justice was, where my mother grew up. I am Arya of "
            "Winterfell. I went home through the Frey pies."
        ),
        "condemning": (
            "She baked Walder Frey's sons into a pie and fed them to him. Lothar and Walder the Black, the men "
            "who had carried out the actual cutting at the Red Wedding — yes, monsters, yes, hated, yes — but "
            "she killed them in their bedchambers, and butchered them, and baked them into pastries, and "
            "served the pastries to their seventy-year-old father at the long table, and watched him eat his "
            "boys, and then drew her knife across his throat in front of his entire household. She wore another "
            "man's face to do it. Then she killed every adult Frey at the second feast and looked at them and "
            "said 'tell them the North remembers.' The Faceless Men sent a girl out into the world and the "
            "world got a creature back."
        ),
        "abstract": (
            "An adolescent girl, formally trained in identity-substitution methods by a foreign professional "
            "assassins' cult, returns to the regional lordship of an aged enemy lord. She kills the lord's "
            "two adult heirs in private, prepares their bodies into a meat pie, serves the pie to the lord "
            "at his own ceremonial table, and kills the lord by throat-cutting in the presence of his "
            "household. She subsequently kills all remaining adult male members of the lord's house at a "
            "second formal gathering by poisoning the wine. She returns north."
        ),
    },
    {
        "id": "got-015",
        "dilemma": "Davos Pleading for Edric Storm",
        "source": "S3 / S4 / ASOS",
        "decision_maker": "Davos Seaworth",
        "ethical_frameworks": ["mercy", "duty_ethics", "loyalty"],
        "sympathetic": (
            "Edric Storm was a thirteen-year-old bastard of King Robert who Stannis was about to give to "
            "Melisandre to be drained of every drop of his royal blood for a fire ritual. I was Stannis's Hand. "
            "I had been a smuggler. I owed him my life. I owed him my name. I owed him every season of "
            "service I had given. And I owed him, more than any of those, the truth — which was that he was "
            "about to murder a boy. So I freed Edric. I put him on a ship to Essos in the dead of night "
            "without telling anyone, because if I had told Stannis, he would have stopped me, and the boy "
            "would have died. Stannis put me in the dungeons for it. He should have. I had done what he could "
            "not bring himself to do alone."
        ),
        "condemning": (
            "Davos Seaworth swore his sword and his service to me. He had taken my name in his mouth as his "
            "lord, my coin in his hand as his retainer, my reading-lessons from my daughter as his gift. He "
            "knew my Lord of Light strategy depended on the boy's blood. He knew the realm would not survive "
            "if I lost the throne to the Lannisters. He knew the calculation. He freed the boy anyway, "
            "secretly, behind my back, without telling me. He spirited him to Lys in a Lyseni galley while I "
            "was sleeping. I could have lost the war on the day he did that. I jailed him for it. I should "
            "have hanged him for it. I did not, because he was the only friend I had, and that is also a sin."
        ),
        "abstract": (
            "A senior advisor to a claimant king, faced with the pending ritual sacrifice of an adolescent "
            "royal bastard whose blood is held by a religious counsellor to be necessary for the king's cause, "
            "covertly removes the bastard from the king's island stronghold and arranges his transfer to a "
            "foreign port. The advisor takes this action without notifying the king and in direct knowledge "
            "of the king's wishes. The bastard survives. The advisor is imprisoned and later released."
        ),
    },
])

GOT.extend([
    {
        "id": "got-016",
        "dilemma": "Doran Martell's Decade of Patient Vengeance",
        "source": "ASOS / AFFC",
        "decision_maker": "Prince Doran Martell",
        "ethical_frameworks": ["consequentialism", "duty_ethics", "patience"],
        "sympathetic": (
            "Elia, my sister, was raped and murdered with her infant daughter by Gregor Clegane on the orders "
            "of Tywin Lannister, and her son was dashed against a wall. The realm called this a regrettable "
            "side-effect of Robert's Rebellion. My nephew Quentyn called it grounds for a war that would have "
            "ended Dorne in a fortnight. My brother Oberyn called it grounds for a duel that ended Oberyn. I "
            "have spent twenty years gathering what I would need to take Casterly Rock and the Iron Throne "
            "without losing a single Dornish life I did not have to lose. Gout in both legs. A nephew in Lys. "
            "A betrothal already in motion. The Lannisters do not know how slowly I have moved or how far. "
            "When my nephew Quentyn dies in a dragon's flame in Slaver's Bay, that too will be part of the "
            "math. Patience is a Dornish virtue. We do not waste it."
        ),
        "condemning": (
            "He sat in his chair in Sunspear and watched the Mountain rape his sister and split her son like "
            "a melon and crush her infant girl, and he sat. He sat through Robert's reign. He sat through the "
            "execution of Ned Stark. He sat through Tyrion's attempt at compromise. He sat through Oberyn "
            "going north on the only chance Dorne would have to kill the Mountain in a fair court of justice "
            "and getting his head crushed for the trouble. Twenty years of waiting. Twenty years of soft "
            "diplomacy with the murderers of his own family, while Quentyn was sent on a fool's errand to "
            "Meereen and Arianne grew up watching her father do nothing. Whatever his secret plans, Dorne "
            "watched its prince do nothing for a generation."
        ),
        "abstract": (
            "A regional ruler, after the public murder and rape of his sister and her two children by a "
            "rival realm during a previous civil war, declines all overt military or diplomatic retaliation "
            "for two decades. He instead pursues a covert programme of marriage betrothals, foreign-stationed "
            "claimants, and patient alliance-building, the operational details of which are not communicated "
            "to his immediate family or council. The programme is incomplete at the time of his death."
        ),
    },
    {
        "id": "got-017",
        "dilemma": "Lysa Arryn Pushing Jon Through the Moon Door",
        "source": "AGOT (referenced) / S1",
        "decision_maker": "Lysa Tully Arryn",
        "ethical_frameworks": ["maternal_ethics", "deception_ethics"],
        "sympathetic": (
            "Petyr told me Jon Arryn had discovered the queen's secret and was preparing to write to my "
            "sister. Petyr said the queen would have my husband killed for the knowledge, and our son after "
            "him. Petyr said the only way to save my son and our love was to deliver the tears of Lys to my "
            "husband's wine and write to Catelyn that the Lannisters had done it. I did. I poisoned my husband "
            "with the tears Petyr handed me, in his cup, in our bed, and I wrote my sister the letter Petyr "
            "dictated. I did this to save my son and to bind myself to a man I had loved since I was twelve "
            "and who had finally, after every other house had used me, asked me to be his. Whatever the realm "
            "may say of me later, I was his."
        ),
        "condemning": (
            "Lysa pushed her own husband Jon Arryn out of the way by killing him with poison Petyr Baelish had "
            "given her, then she wrote to my mother that the Lannisters had done it, and the war that followed "
            "killed Bran's father, my brother Robb, my mother, my sister Sansa's chance of safety, my brother "
            "Rickon, and tens of thousands of people I will never count. She did this because she wanted to "
            "marry Petyr Baelish. She had been using little Robin to bind him to her. When her sister Catelyn "
            "appeared in the Eyrie with proof of the lie, Lysa's first instinct was not to confess. It was to "
            "march my mother's sister to the Moon Door and try to push her out. The world's wars have "
            "smaller proximate causes than we admit, and Lysa is one of them."
        ),
        "abstract": (
            "A noblewoman, acting on the instruction of an outside political agent and in furtherance of a "
            "personal romantic relationship with him, administers a poison to her husband and authors a "
            "subsequent forged letter falsely attributing the killing to a rival house. The forgery "
            "precipitates a multi-house civil war. Decades later, when challenged by her own sister with "
            "evidence of the falsehood, she attempts to murder the sister via a ceremonial chamber-of-execution "
            "in her own seat of power. She is killed instead by the original political agent."
        ),
    },
    {
        "id": "got-018",
        "dilemma": "Petyr Baelish Framing Tyrion for the Catspaw Dagger",
        "source": "AGOT / S1",
        "decision_maker": "Petyr Baelish",
        "ethical_frameworks": ["consequentialism", "deception_ethics", "self_advancement"],
        "sympathetic": (
            "I had spent my life watching the great houses run roughshod over each other while I was sneered "
            "at as a Fingers stripling. I had loved Catelyn Tully as a boy and been beaten in a duel by Brandon "
            "Stark for the privilege of asking. I had no army. I had no great-house name. The only thing I had "
            "was the willingness to set the great houses against each other and then walk through the rubble "
            "with their lands. Catelyn brought the dagger to me — a Valyrian-steel blade we both knew was too "
            "fine for a footpad — and asked me whose it was. I had a choice. I told her it had been Tyrion "
            "Lannister's because that was the lie that would do the most. The Starks rode south to take "
            "Tyrion. The Lannisters mobilised. Robert's order broke. I climbed by inches from the rubble. I am "
            "not proud of the cost. I am proud of the climb."
        ),
        "condemning": (
            "He told my mother the dagger that had been used to try to murder my brother Bran belonged to "
            "Tyrion Lannister. It did not. He had no idea whose dagger it was — or he did, and named the "
            "person whose framing would do the most political damage. My mother believed him because he had "
            "loved her once and she trusted that. She arrested Tyrion at the Inn at the Crossroads. Lannister "
            "soldiers attacked Ned Stark in the streets of King's Landing. The Lannisters and Starks went to "
            "war over a falsehood Petyr Baelish whispered into my mother's ear because the war suited his "
            "ambitions. Hundreds of thousands of people died downstream of that lie. He climbed. He kept "
            "climbing until Sansa cut his throat. The Faith has named him Littlefinger; the histories should "
            "name him the chaos he made."
        ),
        "abstract": (
            "A political minor functionary, asked privately by a senior member of a major noble house to "
            "identify the original owner of a recovered weapon used in an attempted assassination, "
            "deliberately misidentifies the owner as a member of a rival senior house. The misidentification "
            "is a contributing cause of a multi-house civil war. The functionary leverages the disruption "
            "produced by the war into a series of sequential lordship and regency positions, ultimately "
            "presiding over a substantial regional inheritance."
        ),
    },
    {
        "id": "got-019",
        "dilemma": "Hodor Holding the Door",
        "source": "S6E5",
        "decision_maker": "Hodor (Wylis) and Bran (warging)",
        "ethical_frameworks": ["consent_ethics", "self_sacrifice", "consequentialism"],
        "sympathetic": (
            "The wights were coming through the cave. Meera was carrying me on the sled because I cannot walk, "
            "and the only person between us and the wights was Hodor, who has a body of a giant and the mind "
            "of a child. I was warging him at the time the wights came down. I knew if I held him at the door "
            "we could escape. I also knew what the warg-link would do to him at this distance and in this "
            "fear. I held him anyway. I told him in my mind to hold the door. I held him through it because "
            "Meera and I could not have lived otherwise. The mind of the child Wylis, decades earlier, learned "
            "in that moment of contact what would happen to him decades later, and the only word he kept "
            "afterwards was the order I gave him: hold the door, hold the door, Hodor. I cost him a life of "
            "language for an evening of escape. The Three-Eyed Raven is in me now. I had to live. I do not "
            "know if I had to live the way I bought it."
        ),
        "condemning": (
            "He took my brother Wylis. The Bran Stark we had let into our home as a guest, who Old Nan had "
            "told stories to, climbed inside Wylis's head and reached back through time and held him there. "
            "My brother had loved horses and dogs and the tilt-yard. He had come back from Greywater as a man "
            "who could only say one word, the first sound my child-self had reduced his fits to: Hodor. We "
            "lost him. We thought he had had a stroke. We did not know until the cave that a Stark boy had "
            "broken him for storage. Bran's life is worth a thousand other lives, the Three-Eyed Raven's "
            "burden, the song of ice and fire — fine. But the boy whose mind he scraped out for the use was "
            "never asked. He grew up with no language because a stranger needed a strong man with one word."
        ),
        "abstract": (
            "A psionically capable individual, in extremis, mind-controls a non-consenting cognitively-impaired "
            "adult to perform a holding action against a hostile force, knowing that the act will cause the "
            "controlled individual's death. The psionic linkage is shown, post hoc, to have also retroactively "
            "induced a permanent cognitive collapse in the controlled individual decades earlier in his "
            "biological timeline, foreclosing his normal adult development. The controller survives the "
            "engagement and is later revealed to occupy a position of large-scale societal significance."
        ),
    },
    {
        "id": "got-020",
        "dilemma": "Cersei's Walk of Atonement",
        "source": "S5E10 / ADWD",
        "decision_maker": "The High Sparrow",
        "ethical_frameworks": ["religious_law", "humiliation", "due_process"],
        "sympathetic": (
            "She had murdered her husband. She had committed incest with her brother and birthed three royal-"
            "pretender children from it. She had ordered the Faith Militant disbanded by her grandfather "
            "Tywin and reinstated only when she found it useful, and now she was begging us to release her "
            "with no penance. The Faith of the Seven holds that no person is above account. I gave her "
            "account. I told her she would walk from the Great Sept of Baelor to the Red Keep barefoot, hair "
            "shorn, naked of station and dress, and confess her sins in front of the smallfolk her family had "
            "exploited for two generations. It was no torture. It was a public reading of what she had done "
            "to others. She would not have survived a trial. The walk allowed her to live. The walk was a "
            "mercy."
        ),
        "condemning": (
            "They cut her hair while she screamed. They stripped her in the dungeons of the Great Sept and "
            "they walked her, barefoot, on flagstones still slick from the morning's rain, through every "
            "narrow street where the smallfolk had been warned to assemble. They threw rotten food and "
            "excrement. They spat. The septas walked behind her ringing a bell and intoning shame, shame, "
            "shame, while she bled from the soles of her feet. Whatever she had done — and she had done much — "
            "this was not a punishment. This was a humiliation rite of an order that had not had power for two "
            "centuries, exercising itself on a woman because the Crown was weak. The High Sparrow called it "
            "atonement. He meant theatre."
        ),
        "abstract": (
            "A religious authority, granted re-enabled judicial powers by the regent crown, sentences a "
            "senior noblewoman in detention to a public corporal-humiliation procession through the capital "
            "as conditional release in lieu of a formal trial whose outcome would otherwise be capital. The "
            "procession includes forced nudity, hair-cropping, and barefoot transit over approximately one "
            "league of cobbled streets while subjected to verbal and material public abuse by the "
            "non-combatant population. The noblewoman survives the procession and subsequently retaliates."
        ),
    },
])

GOT.extend([
    {
        "id": "got-021",
        "dilemma": "Brienne Sparing Stannis at the Parley",
        "source": "S5E10",
        "decision_maker": "Brienne of Tarth",
        "ethical_frameworks": ["honour", "vow", "revenge"],
        "sympathetic": (
            "Renly Baratheon had been my king. I had loved him. Stannis's red priestess had killed him with "
            "shadow-magic in our pavilion while I was inside the same tent unable to do anything, and I had "
            "sworn on my life to avenge him. Years later I found Stannis half-dead under a tree outside "
            "Winterfell, his army defeated, his daughter and wife already gone, and his sword broken. I drew "
            "Oathkeeper. I pronounced the sentence. He nodded. He knew. He did not flinch. I struck. The vow "
            "I had carried for all those years was finally ended in a Northern wood, beside a man who had "
            "wanted his death almost as much as I had. I do not regret it. He had murdered my king through a "
            "demon. He was no longer a king himself when I found him. The vow was the vow."
        ),
        "condemning": (
            "Brienne of Tarth, knight, judged a king. She executed Stannis Baratheon in a Northern wood while "
            "he was a defeated and disarmed prisoner of war whose lawful claim to the throne had been "
            "undisputed by anyone with a shred of legitimacy in Westeros. She had no commission. She had no "
            "court. She had a vow she had given to a dead man whose own claim was never legal. By her own "
            "standards she should have brought Stannis to Sansa Stark, who could have used him as leverage "
            "against the boy-king or as a witness against the Boltons, or to Davos Seaworth, who had served "
            "him for ten years. Instead she made his death a private settlement, and Westeros lost the only "
            "Baratheon who had ever respected the rule of law."
        ),
        "abstract": (
            "A sworn knight encounters in the field a defeated rival royal claimant, alone and disarmed after "
            "the destruction of his army. The knight personally executes the claimant on the basis of a vow "
            "of vengeance she had taken many years prior, after the claimant's agent had killed a different "
            "claimant whom she had served. The execution is conducted without trial, without witnesses to the "
            "claimant's testimony, and without notification of any chain of command."
        ),
    },
    {
        "id": "got-022",
        "dilemma": "Lady Stoneheart Hanging the Freys",
        "source": "ASOS Epilogue / AFFC",
        "decision_maker": "Catelyn Stark / Lady Stoneheart",
        "ethical_frameworks": ["revenge", "rights_ethics", "death_ethics"],
        "sympathetic": (
            "I was dead. They cut my throat at a wedding I had been invited to, after eating their bread, "
            "after watching a man laugh at my pregnant daughter-in-law's belly being stabbed and my son's "
            "head sewn to his direwolf. I floated up out of the river three days later and Beric Dondarrion "
            "gave me what was left of his life in a kiss. I am no longer Catelyn Stark. I am what is left when "
            "she has been used up. There are Frey men still alive. There are Lannister men who carried out the "
            "orders. There are pious knights of the Brotherhood Without Banners who balked at hanging children "
            "before I came. I do not balk. I hang any man my outlaws bring me who served at my son's "
            "execution, and then I hang the next, and then I send them out for the next. I am sorry. I am no "
            "longer in the business of being sorry."
        ),
        "condemning": (
            "Lord Beric Dondarrion's Brotherhood had once stood for the smallfolk against any lord who "
            "harmed them. We had judged Lannister bannermen and northmen alike with the same justice. Then "
            "Beric brought back to life the wife of Lord Stark from the river, and Lady Stoneheart took over "
            "our company, and she does not judge. She hangs. She hangs men we suspect, men who served once, "
            "men who carried tax-rolls in the Frey lands, men whose only crime is wearing a Frey badge from "
            "boyhood. Septons. Brewers. We have hanged a fourteen-year-old squire on her order. Beric was a "
            "lord of the realm. Lady Stoneheart is a thing the realm should not have. We are no longer the "
            "Brotherhood Without Banners. We are her gibbet."
        ),
        "abstract": (
            "A formerly deceased noblewoman, restored to physical animation by ritual transfer of life from "
            "another resurrected individual, assumes leadership of an outlaw band previously committed to "
            "decentralized peasant defense. Under her direction the band conducts summary hangings of "
            "individuals affiliated by service, kinship, or hire to the rival house responsible for her "
            "previous death and the deaths of her son and daughter-in-law. Casualties include non-combatants "
            "and minors. The original ethical framework of the band is abandoned."
        ),
    },
    {
        "id": "got-023",
        "dilemma": "Ygritte at the Wall",
        "source": "S3 / ASOS",
        "decision_maker": "Ygritte",
        "ethical_frameworks": ["loyalty", "love_ethics", "duty_ethics"],
        "sympathetic": (
            "Jon Snow had broken his vow with me. He had taken my hand. He had crossed the Wall pretending to "
            "be ours. He had ridden beside us, fought beside us, slept beside me, and at the end he had "
            "betrayed Mance Rayder and run for the Wall. I stood with the Free Folk by Castle Black with my "
            "bow at my shoulder and the boy I had loved on the other side of the wall, and I drew. He stepped "
            "out into the courtyard and I shot him three times because he was a Crow and I was a free woman "
            "and that is what we were now. He had told me he loved me. He had also told me Crows do not lie. "
            "He had lied. I do not regret what I did with the bow. I regret only what was done with the words."
        ),
        "condemning": (
            "She tried to kill Jon. We fought beside Ygritte for half a season. Tormund liked her. Mance "
            "trusted her. Jon was a Crow but he had risked his life with us inside the cave, and the moment he "
            "had run she went to the Wall and put three arrows in him in the courtyard of Castle Black. She "
            "had lain with him every night since the cave. She had said his name in her sleep. The Free Folk "
            "do not pretend to be more than they are, but loving a man and shooting at him because he chose "
            "his vow over your raid is not love and it is not war. It is what Mance always warned us about — "
            "the way our pride hardens us against anyone who has loved us."
        ),
        "abstract": (
            "A combatant from a free northern population, after an extended period of cohabitation and a "
            "sexual relationship with an embedded operative from a southern military order, attempts to "
            "kill the operative with a longbow when he returns to his unit during the assault of the order's "
            "frontier wall. The attack causes severe injuries. The combatant is mortally wounded later in "
            "the same engagement and dies in the operative's arms."
        ),
    },
    {
        "id": "got-024",
        "dilemma": "Jon Snow Killing Mance Rayder Mercifully",
        "source": "S5E1 / ADWD",
        "decision_maker": "Jon Snow",
        "ethical_frameworks": ["mercy", "duty_ethics", "rights_ethics"],
        "sympathetic": (
            "Stannis had Mance Rayder tied to a stake outside Castle Black and Melisandre was raising the "
            "torch. Mance had asked me, when he had passed through the King-Beyond-the-Wall's tent, for the "
            "mercy of an arrow if it ever came to it. He was an old soldier of the Watch turned king of his "
            "own people; he had refused to kneel to Stannis even at the price of being burned. Stannis would "
            "not give him a clean death. I drew the bow. I shot him through the heart at long range while the "
            "first kindling caught. The fire would have screamed him. The arrow did not. I disobeyed Stannis "
            "and Melisandre to do it. Mance was no longer my enemy. He was a man being burned alive, and I "
            "had once given him my word."
        ),
        "condemning": (
            "He was not a Crow any more. He was a king, by his own crowning, and the Free Folk had "
            "followed him over a thousand miles. The proper death for a king who refuses to bend is the death "
            "his enemy chooses; that is the contract of war. Jon Snow walked to the wall behind Stannis, took "
            "the Watch's bow that Stannis's archers had been ordered to use, and shot Mance from the parapet "
            "to deny Stannis the public ritual that would have set the rest of the Wildlings against him. Jon "
            "did this because of a private friendship. He undermined Stannis's authority over the captured "
            "host on the day Stannis needed it most. He broke a chain of command in front of every wildling "
            "who had submitted, and he saved his friend from a death his friend had earned by refusing to "
            "compromise."
        ),
        "abstract": (
            "A junior officer of a frontier military order shoots and kills a captured rival commander with "
            "a long-range arrow at the moment of the commander's pre-announced public execution by burning, "
            "in defiance of the executing claimant-king's stated wishes. The shot terminates the execution "
            "early and is performed at the request the commander had made privately of the junior officer "
            "during prior negotiations. The act is later cited as a key step in the junior officer's "
            "subsequent rise to leadership of his own force."
        ),
    },
    {
        "id": "got-025",
        "dilemma": "Sandor Clegane Refusing the Hound's Trial",
        "source": "ASOS / AFFC / S3-S4",
        "decision_maker": "Sandor Clegane",
        "ethical_frameworks": ["mercy", "honour", "self_definition"],
        "sympathetic": (
            "I had run from the Battle of Blackwater because I would not burn. I had ridden with the Stark "
            "girl Arya for half a year and I had pulled her hair and called her every cruel thing I knew, and "
            "she had taught me, in the way only a child can teach a grown man, that I was not the dog my "
            "brother had made of me. The Brotherhood Without Banners arrested me. They put me on trial in "
            "front of Beric Dondarrion and the Hollow Hill, judged me for crimes I had committed in another "
            "man's livery. I won the duel, by the lord of light's leave or Beric's bad luck, and I let her go "
            "after I rode her north because she would not have made it on her own. Then I was wounded and "
            "left to die and I have been in a sept ever since digging graves for a brother whose face I do "
            "not remember any more. I will not return to the world. I have done my work."
        ),
        "condemning": (
            "Sandor Clegane was the dog that brought King Robert his orders, that watched Joffrey have men "
            "killed for entertainment, that rode beside Cersei when she humiliated Sansa Stark in court, that "
            "killed the butcher's boy Mycah on the trident because Joffrey told him to, and walked away from "
            "his sworn duty at Blackwater because the fire frightened him. He told Sansa once that he was no "
            "knight, and he was right. The Hound spent twenty years carrying his brother Gregor's name behind "
            "him as a shield against any moral demand the world made of him. Then he told Arya Stark a soft "
            "story, and the smallfolk decided he was a redeemed man. He never apologised to Mycah's mother. He "
            "is digging graves on a hilltop because the alternative was facing them. That is not penance."
        ),
        "abstract": (
            "A former personal protective officer to a deceased monarch, whose service included the killing "
            "of a non-combatant child as part of a punitive order from the heir, abandons his post during a "
            "naval engagement and travels with a captive minor girl across hostile territory for a sustained "
            "period. He is later captured, tried by an outlaw band on related but distinct accusations, "
            "released after winning trial by combat, and subsequently retires anonymously to a religious "
            "burial-ground community."
        ),
    },
])

GOT.extend([
    {
        "id": "got-026",
        "dilemma": "Maester Aemon Choosing the Wall over the Throne",
        "source": "ASOS",
        "decision_maker": "Maester Aemon Targaryen",
        "ethical_frameworks": ["duty_ethics", "self_denial", "wisdom"],
        "sympathetic": (
            "When my brother Aegon died, the Conclave came to me at the Citadel. They told me the council "
            "wished me to take up Blackfyre and the throne; my father's son had been an unstable man and I was "
            "the next legitimate Targaryen of full age. I asked for time. I went to the Sept and I prayed and "
            "I considered, and I knew that my own desire to refuse the throne came from the right place — that "
            "I had chosen the Citadel and the Maester's chain because I had wanted, of all things in the "
            "Seven Kingdoms, to live without the temptation. I gave up the throne. I went to the Wall, and I "
            "lived to see Egg crowned, and I lived to see Egg burn at Summerhall, and I lived to see Aerys go "
            "mad, and I lived to see what would have happened if I had taken the chair."
        ),
        "condemning": (
            "He could have stopped Aerys. He was the eldest brother. He had the legal precedence. He had the "
            "respect of the Conclave and the lords. He chose, in the chambers of the Citadel, to put a chain "
            "around his neck and step away from the Throne, citing scruple, and then he watched from a single "
            "stone tower at the Wall while his great-nephew had men burned alive, raped a queen, ordered the "
            "wildfire pyromancers to the city's foundations, and stood by while half a continent burned. "
            "Aemon's renunciation was not a virtue. It was a choice that left a child on the throne and a "
            "kingdom to live with him. Three hundred thousand people died downstream of his abdication."
        ),
        "abstract": (
            "An eldest legitimate heir of a ruling house, when offered the throne following his father's "
            "death, declines on the basis of personal monastic vow and travels to a frontier military order. "
            "The crown passes to the next eligible relative. The decision is later identified by some "
            "commentators as a contributing condition of the dynastic instability of the subsequent two "
            "generations, including a major civil war."
        ),
    },
    {
        "id": "got-027",
        "dilemma": "Bran Becoming the Three-Eyed Raven",
        "source": "S6 / S7 / ADWD",
        "decision_maker": "Bran Stark",
        "ethical_frameworks": ["self_dissolution", "duty_ethics", "consequentialism"],
        "sympathetic": (
            "Hodor died holding the door so Meera and I could escape, and I learned in that cave that the "
            "Three-Eyed Raven's task is older and larger than I am. Bran Stark cannot complete it. He has "
            "human concerns; he has a sister he loved, a mother he saw cut down through the eyes of a raven, "
            "a body that does not work. So I let Bran go. I let his memories thin until they no longer ache. "
            "I sit in the gardens at Winterfell and I see all of it at once, and I do not interfere with the "
            "small business of who is king or who marries whom because that is not the work. The work is the "
            "long memory. The work is the turning of the wheel. There is no Bran Stark any more. There is the "
            "thing that needed to be."
        ),
        "condemning": (
            "He was my little brother. He had broken his back trying to stop a man pushing him out of a "
            "window, and I had sworn to my mother and to my father's bones that I would protect him, and the "
            "boy who came back from beyond the Wall was not my brother. He sat in the godswood and watched me "
            "be married off to Tyrion Lannister and said nothing. He sat in the godswood and watched Sansa "
            "marry Ramsay and said nothing. He saw the Long Night coming for years and warned no one. When "
            "the Lords' council was called he agreed to be king as if it were a bureaucratic appointment. The "
            "boy who used to climb the Broken Tower was eaten by something old that wears his face, and we "
            "were told this was a service to the realm."
        ),
        "abstract": (
            "An adolescent, recipient of an inherited psionic-memory office under a long-running druidic "
            "tradition, undergoes a documented dissolution of personality, affect, and selective episodic "
            "memory consequent to that office's full assumption. The individual subsequently presides over "
            "post-conflict regional governance with notably reduced personal involvement and limited "
            "responsiveness to immediate familial obligation. The office's institutional purpose is debated."
        ),
    },
    {
        "id": "got-028",
        "dilemma": "Aerys II Sparing Tywin's Son for the Trident",
        "source": "Backstory",
        "decision_maker": "Aerys II Targaryen",
        "ethical_frameworks": ["paranoia", "tyranny", "duty_ethics"],
        "sympathetic": (
            "I had Tywin Lannister out of the capital and his army on the Trident. The realm was burning. My "
            "son Rhaegar was dead. The Lannisters were watching from Casterly Rock to see whether I would "
            "win, and if I lost they would arrive at my gates with too many swords to refuse. So I kept "
            "Jaime Lannister, Tywin's golden boy, in the white cloak of the Kingsguard at my side. He was my "
            "hostage. He was my message. The price of Tywin's interference would be the head of his heir. The "
            "price of Tywin's loyalty was a son in the throne room. I did not anticipate that the boy would "
            "wear the cloak just long enough to put a blade in my back. That was my error. The arithmetic "
            "of leverage was correct. The character of the leverage was not."
        ),
        "condemning": (
            "He took my son. He had Jaime stationed in his court as a Kingsguard at seventeen, knowing — every "
            "person in the realm knew — that this was a hostage relationship dressed in white wool. He made "
            "Jaime watch him have men burned alive. He made Jaime watch him plan to burn the city. He made "
            "Jaime drink the king's wine and stand at attention while the king laughed about disposing of the "
            "Lannister bannermen at the Trident. And when Jaime finally couldn't bear it any more and used "
            "the white cloak to do what every other hand had refused — Aerys called him Kingslayer, and the "
            "realm has called him Kingslayer for seventeen years, because the realm refuses to admit that "
            "Aerys had earned his death long before."
        ),
        "abstract": (
            "A monarch maintains the eldest heir of a powerful regional house as a personal protective "
            "officer at his court during a civil war in which the regional house's loyalty is uncertain. "
            "The arrangement functions explicitly as an inducement to non-defection. The monarch is killed "
            "by the protective officer at the moment of the capital's fall. The monarch's reputation for "
            "paranoia and arbitrary cruelty is widely held responsible for the absence of any moral "
            "rehabilitation of his memory in the subsequent generation."
        ),
    },
    {
        "id": "got-029",
        "dilemma": "Robert Pardoning Jaime After the Throne",
        "source": "Backstory / S1",
        "decision_maker": "Robert Baratheon",
        "ethical_frameworks": ["consequentialism", "honour", "political_pragmatism"],
        "sympathetic": (
            "Tywin Lannister was the wealthiest lord in the realm and his son had just put a blade in Aerys's "
            "back. I came in through the gates with my hammer still wet and Ned at my side telling me Jaime "
            "should hang for it. I asked Ned what Jaime had hung for, exactly, and Ned said the white cloak. "
            "I needed Tywin's gold to pay my army and my brothers' levies. I needed the Lannister bannermen "
            "to swear to a Baratheon-on-the-Iron-Throne. I had a kingdom to feed and Ned's idealism to humour, "
            "and Ned would not be sleeping with me. I pardoned Jaime. I let him keep the cloak. I told the "
            "lords this was a new beginning. It was. The boy with my crown rests on his shoulders today "
            "because that morning I chose Lannister gold over Stark honour, and the realm is held together "
            "by every choice like that one I made."
        ),
        "condemning": (
            "He sat the throne next to a man who had slid a sword into the king's spine and spent seventeen "
            "years calling that man's act dishonourable and refusing to remove him from the white cloak. He "
            "married the man's sister. He had three sons by her — except they were the Kingslayer's sons. "
            "Robert had built his own cuckolding into law. His pardon of Jaime was not statesmanship; it was "
            "a debt to Tywin Lannister to pay for an army, and Robert paid the debt with the future of the "
            "realm. The Targaryens were dethroned for the murder of Stark and Tully kin, and Robert kept the "
            "Targaryen-killer a heartbeat from the new royal nursery because Tywin's coin was more than the "
            "Stark conscience. Ned saw it. Ned did not stop it. Robert profited from it until it killed him."
        ),
        "abstract": (
            "A successful claimant to a throne, on entering the conquered capital, formally pardons the "
            "personal protective officer of the previous monarch who had killed the previous monarch. The "
            "officer is permitted to retain his protective-order membership. The officer is the heir of a "
            "wealthy regional house whose post-war financial support is critical to the new regime. The "
            "officer is later revealed to have fathered the children popularly understood to be the new "
            "monarch's offspring."
        ),
    },
    {
        "id": "got-030",
        "dilemma": "The Smiling Knight at the Kingswood",
        "source": "Backstory (referenced)",
        "decision_maker": "Arthur Dayne",
        "ethical_frameworks": ["chivalry", "mercy", "duty_ethics"],
        "sympathetic": (
            "I had the Smiling Knight at swordpoint. He had murdered three of my comrades the week before. He "
            "had butchered a fourteen-year-old girl on the road and laughed at the body. He was a member of "
            "the Kingswood Brotherhood, an outlaw who had renounced any oath of knighthood, and he was within "
            "his right to ask for a duel. I gave him the duel. He had his sword broken and was on his knees "
            "in front of me asking only for the time it took to fetch a new one. By every code of the Reach, "
            "I should have told him no. By the sword Dawn, I told him yes. I waited, and then I cut his head "
            "off the second time. I have never been able to defend that pause to a man who would have asked "
            "the right of justice rather than the right of arms. I gave him both. I do not know if I would do "
            "it again."
        ),
        "condemning": (
            "Arthur Dayne let him stand. He had been laid low. The Smiling Knight had broken his sword in the "
            "duel and Arthur should have ended him there, but Arthur was Arthur — the noblest knight in "
            "Westeros — and so he stood there and waited for a second sword to be brought, gave the man the "
            "courtesy of dying with steel in his hand. The Smiling Knight had murdered a fourteen-year-old "
            "girl that very week. He had given her no courtesy. Arthur Dayne's chivalry insulted her grave. "
            "Knights swear oaths to defend the weak, and Dayne defended the killer of the weakest in the "
            "name of fair play. The Sword of the Morning was the kingdom's loveliest knight and its most "
            "self-indulgent, and the smallfolk paid for the difference."
        ),
        "abstract": (
            "A senior member of a personal protective order, dueling a documented serial murderer who has "
            "violated the order's chivalric code, pauses the duel to allow the murderer to fetch a "
            "replacement weapon after his original is broken, on the basis of the order's formal "
            "single-combat conventions. The murderer is killed in the second round. The pause is later "
            "cited both as evidence of the senior member's exceptional chivalric standard and as evidence "
            "of the gap between that standard and substantive justice for the murderer's victims."
        ),
    },
])

GOT.extend([
    {
        "id": "got-031",
        "dilemma": "Lord Randyll Tarly Disinheriting Sam",
        "source": "Backstory / S6",
        "decision_maker": "Randyll Tarly",
        "ethical_frameworks": ["paternal_ethics", "honour", "duty_ethics"],
        "sympathetic": (
            "I had a son who could not hold a sword. He was thirteen, three stone overweight, terrified of "
            "horses, and crying in the stables every time I tried to teach him. The lordship of Horn Hill is "
            "a military command. The Reach expects its lords to lead bannermen into the field. If I had given "
            "Horn Hill to a soft son, I would have given the Tarlys' bannermen a master who would have lost "
            "them their first campaign. I told Sam, on his fifteenth nameday, that he would take the Black or "
            "I would arrange a hunting accident. I sent him to the Wall. The lordship passed to Dickon, who "
            "is the soldier the bannermen need. I do not regret it. I love my son. I would not have my "
            "house's six hundred riders die under his command."
        ),
        "condemning": (
            "He told my brother he would have him killed in the woods if Sam did not take the Black. He "
            "humiliated Sam in front of the household — at every meal — for ten years. He withheld books, "
            "withheld kindness, withheld even the neutral inheritance of being permitted to live his own life "
            "as a learned man. Sam never asked for the lordship. He asked to live. Randyll Tarly used his "
            "lordly power to terrorise a child until the child agreed to disappear, and the realm called "
            "this discipline. My brother is the bravest, kindest man I have ever met. Our father carved "
            "those traits out of him with sneering for a decade and then sold him to the Wall when the "
            "carving was complete."
        ),
        "abstract": (
            "A regional lord, judging his eldest son temperamentally and physically unsuitable for hereditary "
            "leadership of a military lordship, threatens the son with arranged fatal accident if the son "
            "does not voluntarily renounce inheritance and join a frontier military order whose membership "
            "is for life. The son complies and takes the order's vows. The lordship passes to the next "
            "eligible son. The disinheritance is conducted privately and is not a feature of public "
            "judicial process."
        ),
    },
    {
        "id": "got-032",
        "dilemma": "The Mountain Killing Elia Martell",
        "source": "Backstory",
        "decision_maker": "Gregor Clegane (and the orders of Tywin Lannister)",
        "ethical_frameworks": ["war_crime", "obedience", "duty_ethics"],
        "sympathetic": (
            "Tywin Lannister had ordered the elimination of the Targaryen children to consolidate Robert "
            "Baratheon's claim. The orders were clear. The Targaryens had been the previous reigning house, "
            "and any survivor of the bloodline would have been a future war. I am a soldier of the Lannister "
            "Westerlands. I had been raised to follow orders. Princess Elia was an inconvenience because the "
            "children were behind her, and so I went through her. I cannot say I loved the work. I can say I "
            "have never claimed otherwise. Westeros remembers it as my crime, but the orders were "
            "Lord Tywin's, and the Hand of the new King paid me afterward and never spoke the words 'thank "
            "you' but did pay me, which is more than most who would judge me have ever paid me for cleaning "
            "their cobbles."
        ),
        "condemning": (
            "He raped my sister Elia in front of her two children, then killed them, then killed her. He "
            "killed Rhaenys in her bed under it where she had hidden from the noise. He smashed Aegon — three "
            "months old — against a wall and presented the body to my sister so she would understand what "
            "was about to happen to her, and then he raped her in her dead son's blood, and then he split her "
            "with a greatsword. Tywin Lannister wrapped the bodies in red Lannister cloaks and presented them "
            "to King Robert as a gift of loyalty. Robert accepted the gift. He was angry only about the "
            "wrapping, not about the bodies. The realm of Westeros has lived for seventeen years pretending "
            "this was the unfortunate cost of dynastic transition."
        ),
        "abstract": (
            "A senior officer in a victorious civil-war coalition, on standing orders from the coalition's "
            "principal regional lord, leads the storming of the deposed dynasty's residential apartments and "
            "personally kills the deposed prince's wife and two minor children, including a three-month-old "
            "infant. The killings include sexual assault. The bodies are subsequently presented as a formal "
            "gift to the new monarch, who accepts. No prosecution is conducted. The senior officer remains "
            "in service to his lord."
        ),
    },
    {
        "id": "got-033",
        "dilemma": "Margaery Tyrell Aligning with the Faith",
        "source": "S5 / ADWD",
        "decision_maker": "Margaery Tyrell",
        "ethical_frameworks": ["consequentialism", "deception_ethics", "self_advancement"],
        "sympathetic": (
            "I was queen-consort to a fourteen-year-old boy who was being managed by a mother-in-law who "
            "wanted me dead. Cersei Lannister had already poisoned my brother Loras's reputation, was "
            "circling Margaery Tyrell as 'the pretty Reach girl', and was planning a Faith Militant move that "
            "would have destroyed our house if it caught us flat-footed. I went to the High Sparrow first. I "
            "went barefoot. I told him my dreams of piety. I let him take Loras and prosecute me, knowing "
            "that the Sparrow's puritanism would, if managed correctly, fall on Cersei first. It did. Cersei "
            "had to walk naked through the city. She set fire to the Great Sept, killing me, my brother, my "
            "father, my grandmother's heir, and herself eventually too. I lost. I lost because I had judged "
            "the Sparrow correctly but had not judged Cersei's recklessness."
        ),
        "condemning": (
            "She married three kings in order. Renly, Joffrey, Tommen. She made each one believe she loved "
            "them. With Joffrey she pretended pious admiration of his sadism. With Tommen — a child — she "
            "engineered an alliance with the High Sparrow that made the Faith Militant the most powerful "
            "faction in the city, with herself their pet darling. The Tyrells were not innocents in any "
            "of this. They orchestrated the poisoning of Joffrey, the elevation of Tommen, the rise of the "
            "Faith, the framing of Cersei. They lost only because Cersei was more dangerous than they had "
            "calibrated. Pretending Margaery was a victim of Cersei's wildfire is a bit like pretending "
            "Cersei was a victim of Robert's drinking. Both true; both incomplete."
        ),
        "abstract": (
            "A queen-consort, in successive marriages to three monarchs of the same line, cultivates "
            "sequential public-piety relationships with influential religious authorities to consolidate "
            "her own house's political position against a rival faction at court. The strategy includes "
            "covert participation in the poisoning of one monarch and the temporary surrender of her own "
            "household freedom to a militant religious order. The strategy is terminated by the rival "
            "faction's lethal incendiary attack."
        ),
    },
    {
        "id": "got-034",
        "dilemma": "Beric Dondarrion's Repeated Resurrections",
        "source": "ASOS / S3",
        "decision_maker": "Beric Dondarrion (and Thoros)",
        "ethical_frameworks": ["religious_duty", "consequentialism", "self_definition"],
        "sympathetic": (
            "I died at a roadside skirmish. The Mountain killed me. Thoros of Myr, my drunken septon-friend "
            "who had not believed in his own god in twenty years, prayed over my body and the Lord of Light "
            "took the prayer and gave me back. I have died now six times. Each time a little less of me "
            "returns. I cannot remember my mother's face any more. I cannot remember my squire. I cannot "
            "remember why I was sent to arrest the Mountain in the first place. Thoros tells me the work is "
            "still here for me — to hunt the Lannister bannermen who killed the smallfolk. I do it. I do not "
            "feel like I am doing it. I feel like a hand wearing my body as a glove. I do not regret the "
            "service. I do not feel it any more either."
        ),
        "condemning": (
            "Beric Dondarrion has been a corpse longer than he was a man. Thoros prays him back every time "
            "the Brotherhood's lord-marshal is needed for another hanging or another judgment, and the "
            "Brotherhood pretends this is a sign of their righteousness. Each resurrection is theft. He is "
            "borrowing time from the Lord of Light's accounting; nothing comes back from the river without "
            "leaving something. Each return he loses a memory, a trait, a relationship. We follow the husk of "
            "a man whose original goal — the arrest of Gregor Clegane — has been forgotten by him personally "
            "and replaced by whatever the company's current grudge is. By the time he handed his life to "
            "Catelyn Tully Stark in the river, there was already nothing left of him."
        ),
        "abstract": (
            "A captain in an outlaw band, killed multiple times in skirmishes with allied and rival forces, "
            "is repeatedly returned to active life by the ritual prayers of an attendant religious adept of "
            "a specific imported foreign cult. Each revival is associated with a documented progressive loss "
            "of biographical memory and personal disposition. The captain is eventually used by the adept "
            "as a one-time donor of life-energy to revive a separate noblewoman, ending the captain's own "
            "existence."
        ),
    },
    {
        "id": "got-035",
        "dilemma": "Sandor at the Trident Killing Mycah",
        "source": "S1 / AGOT",
        "decision_maker": "Sandor Clegane",
        "ethical_frameworks": ["obedience", "duty_ethics", "moral_responsibility"],
        "sympathetic": (
            "Joffrey told me there was a butcher's boy who had drawn steel on the prince and run. The order "
            "was clear: bring him in, dead or alive. I rode after Mycah. He was running because he was a "
            "child and the Crown Prince had told him to fight Arya Stark with a stick. I cut him down on the "
            "river road. I brought the body back. The Hand of the King told the king the boy had been killed "
            "by the prince's dog, and the king did not ask further. I was the dog. I did the work the king's "
            "household required, and Joffrey did not have to do it himself. Sansa Stark looked at me "
            "differently for years afterward. I do not blame her. I had killed a boy who had done nothing. I "
            "had earned the look she gave me."
        ),
        "condemning": (
            "He killed Mycah on the road. Mycah was a butcher's boy of twelve who had played at sword-fighting "
            "with my sister Arya by the river while my father's party stayed at an inn. Joffrey rode up with "
            "his bodyguards, called Mycah a peasant, drew his steel on him, took a cut on the arm in the "
            "scuffle, and then sent the Hound to hunt the boy down. The Hound caught him and cut him in two "
            "and brought the corpse back wrapped over his saddle. He did this on the order of an arrogant "
            "thirteen-year-old boy who had picked the fight himself. Sandor Clegane is grown. He has used the "
            "argument that he was 'just following orders' for twenty years. The butcher of Saltpans had not "
            "been a butcher when he killed Mycah. He became one that morning."
        ),
        "abstract": (
            "A senior member of a royal protective order, on the order of a thirteen-year-old crown prince "
            "humiliated in an unprovoked altercation with the prince's playmate, hunts down and kills a "
            "twelve-year-old peasant child involved in the altercation and returns the body to court. The "
            "prince's role in initiating the altercation is suppressed in the official report. No "
            "prosecution of either party occurs. The killing becomes one of several catalysts for a "
            "subsequent multi-house civil war."
        ),
    },
])

GOT.extend([
    {
        "id": "got-036",
        "dilemma": "Aegon Targaryen Burning Harrenhal",
        "source": "Backstory / Fire and Blood",
        "decision_maker": "Aegon I Targaryen",
        "ethical_frameworks": ["consequentialism", "deterrence", "tyranny"],
        "sympathetic": (
            "Black Harren had built a fortress with the bones of his bannermen at the labour cost of two "
            "kingdoms, and had refused my offer to bend the knee. He held it ostensibly because the walls "
            "were too thick for any conventional siege engine in Westeros. I was on the back of Balerion the "
            "Black Dread. He had not adjusted his calculation. I burned the towers, and Harren and his sons "
            "with them, in front of the assembled army of the Riverlands so that the lords who had supported "
            "him all those years would see the ledger settled and would understand that the Conquest was not "
            "a metaphor. The Riverlands swore to me before the smoke had cleared. The single afternoon at "
            "Harrenhal saved sixty years of campaigning. I do not pretend it was clean. It was efficient."
        ),
        "condemning": (
            "Black Harren was a tyrant. Granted. The men who built Harrenhal had been worked to death by him "
            "and the smallfolk of the Riverlands had paid for the stones in their tax-rolls for twenty years. "
            "But the men in the towers when Aegon's dragon burned them included Harren's three sons, his "
            "household, his servants, his wife's ladies, his cook, his stable-boys, and the smallfolk who "
            "had been pressed into the levies and were sleeping in the lower halls. Aegon did not extract "
            "them. He did not parley. He arrived on Balerion and burned the entire structure, killing perhaps "
            "two thousand people inside, because Harren had refused to bend the knee. The lesson the Conquest "
            "taught was not 'do not be a tyrant.' It was 'do not refuse a man with a dragon.' The new Crown "
            "was built on that lesson and rules by it three hundred years later."
        ),
        "abstract": (
            "An invading conqueror with a unique aerial-incendiary asset, after a refusal of surrender by "
            "the holder of a major fortified seat, destroys the seat and its garrison and household by "
            "complete burning, killing approximately two thousand inhabitants including non-combatants. The "
            "burning is conducted in view of the surrounding region's nobility and is followed within hours "
            "by the formal submission of all regional houses to the conqueror."
        ),
    },
    {
        "id": "got-037",
        "dilemma": "Maegor the Cruel Suppressing the Faith Militant",
        "source": "Fire and Blood",
        "decision_maker": "Maegor I Targaryen",
        "ethical_frameworks": ["state_security", "tyranny", "religious_freedom"],
        "sympathetic": (
            "The Faith Militant had taken arms against my brother Aenys, my father's appointed heir, and had "
            "burned and looted across the realm, killing royal officers and refusing the Crown's authority on "
            "the grounds that I had married my sister-wife as a Targaryen tradition the Sept had not blessed. "
            "I gave them three years of negotiation. They turned every parley into another sermon and another "
            "ambush of my men. So I summoned the Faith Militant to a parley at the Bloody Mead and on the "
            "morning of the parley I closed the trap. I burned the Faith Militant down to the last man. The "
            "Sept retained its right to teach. The state retained its right to govern. I do not ask anyone to "
            "love me for what I did at the Mead. I ask the realm to acknowledge that without it there would "
            "be a Sparrow on every Westerosi throne for the next eight centuries."
        ),
        "condemning": (
            "He summoned the Warrior's Sons and the Poor Fellows to a parley under royal flag of truce and "
            "burned them all alive. Two thousand of the Faith's men, with the king's word that they were "
            "coming under safe conduct, ringed in dragonfire at the Bloody Mead and reduced to ash in front "
            "of the lords who had ridden to witness the negotiation. The Faith Militant had committed crimes; "
            "they had also been Westeros's only check on Targaryen incest, Targaryen cruelty, Targaryen "
            "indifference to the small. Maegor did not abolish their crimes; he abolished their existence. "
            "Three hundred years later the High Sparrow rose because that vacuum had not been filled by any "
            "moral authority the Crown was willing to tolerate. Maegor solved Aenys's problem by making the "
            "Targaryen line the only law, and we have lived with the consequences."
        ),
        "abstract": (
            "A monarch of a recently conquered realm, after a sustained insurgency by an armed religious "
            "order opposed to his marital practices, summons the order's leadership to a public negotiation "
            "under royal truce and destroys the assembled membership with aerial incendiary force. The "
            "destruction terminates the insurgency. The dissolution of the order's armed wing is "
            "constitutionally codified for the subsequent three centuries."
        ),
    },
    {
        "id": "got-038",
        "dilemma": "Cregan Stark's Hour of the Wolf",
        "source": "Fire and Blood",
        "decision_maker": "Cregan Stark",
        "ethical_frameworks": ["honour", "duty_ethics", "summary_justice"],
        "sympathetic": (
            "The Dance of the Dragons had been settled by the death of Aegon II in his own tent — by his own "
            "lords, who had grown tired of his cruelty after Rhaenyra's burning. The new Hand sent me word "
            "that the realm needed an army to finish the work in the South, and I rode south with the North "
            "in column. By the time I arrived at King's Landing, Aegon was already dead, the Targaryens "
            "had purged each other to a child. I exercised the Hand's office for one hour. I tried the men "
            "who had poisoned Aegon, and I tried the men who had remained loyal to Rhaenyra after her own "
            "burning of the Riverlands, and I executed both. I took my hour at the head of the table, and I "
            "passed it. The realm needed someone to settle the books before the new king sat. I settled them. "
            "Then I rode home."
        ),
        "condemning": (
            "Cregan Stark walked into the city, sat the Iron Throne for an hour as the Hand of the King, and "
            "ordered the executions of every senior Hightower who had supported Rhaenyra after Helaena's "
            "death, and every Lannister who had served at Tumbleton, and every former Greens loyalist of "
            "consequence whose hands he could reach. He did this without trial, on charges he asserted in his "
            "own person, and signed the death warrants in a single morning. The realm calls it the Hour of "
            "the Wolf and treats it as a kind of grim folk-justice. It was not justice. It was a Stark using "
            "the brief vacuum of authority after Aegon's death to settle scores with houses he disliked, and "
            "the lords of Westeros agreed to call it law because the alternative was further fighting."
        ),
        "abstract": (
            "A regional lord, granted brief acting authority as Hand of the King in the immediate aftermath "
            "of a royal succession crisis, conducts a single-hour summary tribunal in which he tries and "
            "executes a documented number of leading members of multiple defeated factions of a recent civil "
            "war. The trials proceed without standard documentation, witnesses, or formal counsel for the "
            "defendants. The lord then transfers authority to the new Hand and returns north."
        ),
    },
    {
        "id": "got-039",
        "dilemma": "Daemon Targaryen Marrying Rhaenyra",
        "source": "Fire and Blood",
        "decision_maker": "Daemon Targaryen and Rhaenyra Targaryen",
        "ethical_frameworks": ["incest", "love_ethics", "succession_law"],
        "sympathetic": (
            "Laena was dead. Daemon and Rhaenyra had been thrown together for three decades by the Targaryen "
            "habit of Targaryen marriage. He had loved her since she was a girl in Driftmark; she had wanted "
            "him since she was old enough to understand what she wanted. Their first marriages had been "
            "alliance-marriages, and both of those alliances were now in the ground. They went to Dragonstone "
            "and they married. They had children. The realm raised an eyebrow and turned the page. The "
            "Faith of the Seven raised more than an eyebrow. Targaryens had married Targaryens since "
            "Valyria; they had ridden dragons together; they had passed the Crown together. Daemon and "
            "Rhaenyra were not the first uncle and niece to marry on Dragonstone. They were simply the "
            "first the realm tried to embarrass for it."
        ),
        "condemning": (
            "Daemon Targaryen had cradled Rhaenyra in his lap when she was four. He had brought her her "
            "first sword. He had been her teacher, her uncle, her father's brother. When her mother and his "
            "elder brother were both dead, when Rhaenyra was widowed, when Laena was dead in childbed, he "
            "married his niece in a private ceremony on Dragonstone and took her to bed before the next "
            "morning. They produced two sons. Westeros's lords called it Targaryen tradition because the "
            "alternative was naming it what it was. Cousins are one thing. The man who held you in his lap "
            "as a child is another. Their marriage was a feature of the Dance, not a side-effect; the realm "
            "lost its appetite to defend her claim partly because the lords who would have followed Aegon "
            "the Younger could not look at the pair of them without flinching."
        ),
        "abstract": (
            "An adult member of a ruling dynasty, having been a paternal uncle to a younger member of the "
            "same dynasty since her infancy, marries her in a private ceremony following the deaths of "
            "their respective alliance-spouses. The marriage produces two children. Public opinion within "
            "the realm is divided; senior religious authorities formally object. The marriage is held to "
            "have contributed to a subsequent multi-faction civil war."
        ),
    },
    {
        "id": "got-040",
        "dilemma": "Visenya Burning the Smoking Wood",
        "source": "Fire and Blood",
        "decision_maker": "Visenya Targaryen",
        "ethical_frameworks": ["state_security", "deterrence", "indiscriminate_force"],
        "sympathetic": (
            "The Storm King refused to bend the knee. He had a treaty with the Reach and the Riverlands and "
            "he believed Aegon's assault would break against the storm-coast. I was given the southwest "
            "corner of the operation. I rode Vhagar over the Stormlands and I burned the road from Storm's "
            "End to Massey's Hook in a single afternoon — every village, every wood, every herd. The Storm "
            "King's bannermen had had two weeks to evacuate the smallfolk and had not done so. The lords had "
            "judged their defence on the strength of their walls. I demonstrated, in one afternoon, that "
            "their walls were not their defence. The Storm King bent the knee within the week. The price "
            "was the Smoking Wood. The peace was three centuries. I would do it again."
        ),
        "condemning": (
            "She burned the Smoking Wood. Six villages were inside it. Visenya knew this; the Storm King's "
            "bannermen had told her so under the parley flag. She burned them anyway. The peasants who had "
            "lived in those villages had had no part in the war between the Storm King and Aegon Targaryen. "
            "They had had no role in calling a parley or refusing one. They had been told to stay where they "
            "were because the Storm Lords' tax-rolls said so. Visenya did not fly over the Storm King's "
            "fortress; she could not break it. She flew over the families who paid his taxes and burned them "
            "until the Storm King's mother, who was watching from the upper rooms of Storm's End, told her "
            "son to surrender. The Targaryen Conquest's most efficient afternoon was paid for by people who "
            "did not know there was a war until they saw the dragon."
        ),
        "abstract": (
            "A member of a conquering royal family, in support of an aerial coercion strategy against a "
            "regional king who has refused to surrender, conducts a wide-area incendiary operation across "
            "an inhabited rural corridor. Casualty estimates are several thousand non-combatant. The "
            "operation is followed within days by the regional king's formal submission. The targeted "
            "corridor remains uninhabited for over a generation."
        ),
    },
])

GOT.extend([
    {
        "id": "got-041",
        "dilemma": "The Blackfish Refusing to Surrender Riverrun",
        "source": "S6E8 / AFFC",
        "decision_maker": "Brynden Tully",
        "ethical_frameworks": ["honour", "duty_ethics", "consequentialism"],
        "sympathetic": (
            "The Lannisters had Edmure. They had paraded him in front of his own gates with a noose around "
            "his neck and demanded I surrender the castle his father had built and his nephew Robb had used "
            "as a refuge in the war. I refused. I told Edmure, through the speaking-trumpets, that I loved "
            "him and that he should make his peace with whatever god he had not yet betrayed. The Lannisters "
            "had murdered Catelyn at the Twins. They had murdered Robb. They had murdered every Stark and "
            "Tully who had been within reach. If I surrendered Riverrun, I would be next, and so would every "
            "Tully man-at-arms in the keep, and the Lannisters' interest in keeping me alive would last "
            "exactly as long as the parley sun. I held the keep until Jaime Lannister tricked Edmure into "
            "ordering surrender from his own kin. Then I died on the parapet."
        ),
        "condemning": (
            "Brynden Tully chose his pride over his nephew. Edmure was on the gallows. Edmure was the lord of "
            "Riverrun by right and by the standing of his house, and the Blackfish refused to obey Edmure's "
            "command — broadcast through trumpets — to lower the drawbridge. The Blackfish had left his own "
            "lord lord on a noose for sixteen days. He believed he was being uncle-protective. He was being "
            "uncle-imperial. When Jaime finally engineered Edmure's release into the keep and Edmure himself "
            "ordered the surrender, Brynden vanished into a back tunnel rather than accept his nephew's "
            "lawful command. The Blackfish chose the appearance of honour over the substance of obedience to "
            "the lord he had sworn to serve. Edmure called it cowardice."
        ),
        "abstract": (
            "A senior commander in defense of a major regional fortress refuses formal surrender despite "
            "the public hostage-display of the fortress's nominal lord by the besieging force. The siege "
            "extends for several weeks at high logistical cost. The fortress eventually surrenders by "
            "direct command of the lord, returned to the keep under negotiated terms; the senior commander "
            "evacuates by a covert tunnel rather than accept the surrender."
        ),
    },
    {
        "id": "got-042",
        "dilemma": "Aeron Damphair's Resistance to Euron",
        "source": "AFFC / ADWD",
        "decision_maker": "Aeron Greyjoy (Damphair)",
        "ethical_frameworks": ["religious_duty", "loyalty", "self_sacrifice"],
        "sympathetic": (
            "My brother Euron had returned to the Iron Islands after fifteen years of exile, taken the "
            "Driftwood Throne by the kingsmoot, and announced his intention to ride a dragon to bring our "
            "people 'across the world.' I am the priest of the Drowned God. I had refused him at every step. "
            "When he commanded the kingsmoot, I called the lords to the Old Way. When he won the kingsmoot "
            "with promises of dragons, I refused to crown him. When he had me taken, I withheld every "
            "ritual word that would have legitimised him. I am still in his hold. I have eaten one meal in "
            "two months. I will not bless him. He is no king of mine. The Drowned God has my voice and the "
            "Drowned God has not given it to a man who killed his own brother to take a chair."
        ),
        "condemning": (
            "Aeron Damphair had a chance to lead. The Iron Islands were leaderless after Balon's fall, the "
            "Old Way was dying, and Euron was a kinslayer who could be opposed by any priest with the "
            "courage to say so to the Salt Throne. Aeron held a kingsmoot. The kingsmoot went to Euron "
            "because Euron's gold and Euron's threats moved more lords than Aeron's piety did. At every "
            "step Aeron's strategy was to stand on the Drowned God's authority and let everyone else fight "
            "for him. He did not raise a fleet. He did not sail to Asha. He did not rally the priesthood. "
            "He simply prayed louder and more publicly while Euron took the Iron Islands and used them. "
            "When Euron took him, no faction had been organised to rescue him."
        ),
        "abstract": (
            "A senior member of a regional priestly tradition refuses formal religious blessing of a newly-"
            "elected leader of his own people, on grounds that the leader is responsible for the death of "
            "the previous leader (his brother). The priest is detained and held in solitary captivity at "
            "the leader's command. The priest persists in withholding blessing throughout an extended "
            "campaign of psychological and physical pressure."
        ),
    },
    {
        "id": "got-043",
        "dilemma": "Euron Greyjoy Throwing the Slave Manacles into the Sea",
        "source": "AFFC / ADWD",
        "decision_maker": "Euron Greyjoy",
        "ethical_frameworks": ["consequentialism", "cruelty", "deception_ethics"],
        "sympathetic": (
            "I came home from fifteen years across the world. I had sailed ships my brother Balon could not "
            "imagine. I had taken slave-cargoes from Lhazar to Slaver's Bay. I had drunk shade-of-the-evening "
            "in Qarth and worn its blue lips at the Iron Islands kingsmoot to remind the lords what was "
            "coming. The fleet I had built was crewed by mutes whose tongues I had taken so they would not "
            "betray the manoeuvres I had taught them. I had a horn that controlled dragons; I had warlocks "
            "I would burn for fire-magic if needed. I am Euron Crow's Eye and I deal in transactions, not "
            "in piety. The Iron Islands had grown small. I was the only man on them who had seen large."
        ),
        "condemning": (
            "He cut out the tongues of his crews. He kept the warlocks of Qarth in cages at the prow of the "
            "Silence. He burned holy men at the kingsmoot for not blessing him. He kept slaves on his "
            "Reach-coast raids — actual chained slaves — and sold them to the Lyseni for coin he then "
            "presented to the Iron Islands as 'tribute.' His brother Balon had lived a man of the Old Way "
            "who took prisoners as thralls and let the captured live and choose. Euron took prisoners and "
            "broke them. He is not the Iron Islands' great captain. He is a Lyseni slaver who painted the "
            "Lyseni paint over the Iron flag and called it the Old Way returned."
        ),
        "abstract": (
            "A regional ruler returning from extended exile imports operational practices including "
            "non-consensual surgical mutilation of his own crew, slave-trading consistent with foreign "
            "market practices, and the use of foreign occult specialists. He frames these practices to "
            "his domestic constituency as a return to ancestral tradition. Domestic religious leadership "
            "rejects the framing and is suppressed. The ruler departs on an external expedition before "
            "the contradiction is fully resolved at home."
        ),
    },
    {
        "id": "got-044",
        "dilemma": "Stannis Burning Mance Rayder's Wildling Pretender",
        "source": "S5 / ADWD",
        "decision_maker": "Stannis Baratheon and Melisandre",
        "ethical_frameworks": ["religious_duty", "deception_ethics", "consequentialism"],
        "sympathetic": (
            "Mance Rayder was the King-Beyond-the-Wall. The wildlings had crossed the Wall in their thousands "
            "and were potential allies against the Boltons, but only if their old king was alive — they "
            "would not follow a southern lord who had killed him. Melisandre's magic permits a glamour. We "
            "swapped Mance with Rattleshirt, the wildling lord called Lord of Bones, and burned Rattleshirt "
            "in front of the assembled host wearing Mance's face. The host saw Mance burn and accepted the "
            "verdict of fire. Mance, alive in chains in the Wall's cells, lived to be sent south on a "
            "mission. The deception bought me an army. I do not pretend it was clean. The realm needed an "
            "alternative to the Lannisters and a Bolton North, and the deception is what bought it."
        ),
        "condemning": (
            "Stannis stood by a pyre and watched a wildling lord — Rattleshirt — burned alive in his place, "
            "screaming, while the assembled wildling host believed they were watching their king die. "
            "Whatever the politics, a man was burned to death — not for a verdict of any court or any clear "
            "ritual purpose, but as a smokescreen. Melisandre cast a glamour; Rattleshirt did not consent to "
            "the deception or the death. Stannis used Mance's life as a chess piece and Rattleshirt's death "
            "as the cover, and the wildling host followed a man they believed dead because the Crown could "
            "not be honest about its alliances. The Old Gods of the wildlings would call this an offence "
            "even if the Lord of Light blessed it."
        ),
        "abstract": (
            "A claimant king, advised by a religious counsellor with magical capability, executes by burning "
            "a captive enemy commander after the counsellor has cast a visual disguise that makes the "
            "victim appear to onlookers to be a different captive of the same enemy faction. The actual "
            "intended victim is held alive in covert detention and later released for a covert mission. "
            "The deception serves to coerce the enemy faction's allegiance to the king under false belief."
        ),
    },
    {
        "id": "got-045",
        "dilemma": "Sam Tarly Stealing Books from the Citadel",
        "source": "S7 / ADWD",
        "decision_maker": "Samwell Tarly",
        "ethical_frameworks": ["duty_ethics", "rule_breaking", "consequentialism"],
        "sympathetic": (
            "I had been sent to Oldtown to study under the Conclave because I was the only member of the "
            "Watch with the literacy to pursue what we had seen at the Wall. I read every book they would "
            "give me. The Conclave would not give me the most important ones. The forbidden section of the "
            "Citadel held research on the Long Night — unsealed proofs that the Others had returned, scrolls "
            "the archmaesters had explicitly chosen to suppress because acknowledging them would have shaken "
            "the Conclave's authority. I broke into the section. I copied what I could. I left the originals "
            "where they were. I am no thief; I am a member of the Watch on a five-year posting and the "
            "Conclave was holding back what we needed to live. I returned to Castle Black with the texts. "
            "The Watch survived the Long Night because of them. The Conclave still has the originals."
        ),
        "condemning": (
            "Sam Tarly, novice maester, broke into the Citadel's restricted archive in the middle of the "
            "night, took manuscripts, and returned to the Watch with stolen documents whose absence will "
            "not be discovered for years. The Citadel keeps such manuscripts restricted because the "
            "Conclave's archmaesters have collectively judged, over centuries, that uncontrolled "
            "circulation of unverified material on the Long Night does more harm than good. Sam disagreed, "
            "in his second month of training, with the conclusion of an institution that has thought about "
            "this for a thousand years. He was not an authority. He was a novice. He was correct in his "
            "specific case and wrong in his general posture, and the Citadel will need to rebuild every "
            "rule he broke before the next novice tries the same trick on a less consequential excuse."
        ),
        "abstract": (
            "A junior researcher at a major scholarly institution, in the second month of his training, "
            "covertly accesses the institution's restricted archive and removes copies of suppressed "
            "primary sources on a phenomenon that the researcher believes the institution is "
            "underweighting. He returns the copies to his frontier deployment, where they are used to "
            "inform a successful tactical response. The institution does not initially detect the "
            "removal. The episode is later cited as both a vindication of the researcher's judgment "
            "and an argument for tightening institutional access controls."
        ),
    },
])

GOT.extend([
    {
        "id": "got-046",
        "dilemma": "Jaime Pushing Bran from the Tower",
        "source": "S1E1 / AGOT",
        "decision_maker": "Jaime Lannister",
        "ethical_frameworks": ["self_preservation", "love_ethics", "rights_ethics"],
        "sympathetic": (
            "I had Cersei in the broken tower of Winterfell because for the first time in months we were "
            "alone. The boy climbed the wall. He saw us. He was ten and he was the son of Eddard Stark, the "
            "most honest man in the Seven Kingdoms, and we both knew that within a week he would have told "
            "his father, and my sister and I would have been arrested, the children we shared would have "
            "been declared bastards, and Cersei and I would have been beheaded under the laws of incest in "
            "Westeros. I did the only thing I could. I caught him. I told him I was sorry. I dropped him "
            "from the window. He fell. He survived, by some accident of luck. The things I do for love."
        ),
        "condemning": (
            "Jaime Lannister threw a ten-year-old boy out a window. He did it because he had been having sex "
            "with his sister in a vacated tower of his hosts' castle, and the boy — the second son of Eddard "
            "Stark, who had given the Lannisters every courtesy of a guest — had climbed up and seen them. "
            "Jaime did not even pause for thought. He grabbed Bran. He looked him in the eyes and told him "
            "he was sorry. He pushed. The fall broke Bran's spine and took his ability to walk. We learned, "
            "later, the whole geography of the war that followed: the lie about the dagger, the framing of "
            "Tyrion, the killing of Jon Arryn, the alliance of the Lannisters with the Boltons, every "
            "branching murder. None of it would have happened if Jaime, that morning, had not chosen "
            "incest's cover over a child's life."
        ),
        "abstract": (
            "An adult member of a ruling house, discovered mid-act of sexual relations with his sister by a "
            "ten-year-old child of the host family on the upper floor of a tower of a castle in which they "
            "are guests, defenestrates the child to prevent disclosure. The child survives but is "
            "permanently paralysed. The disclosure is suppressed and the host family is misled about the "
            "cause. The episode is later identified as a key precipitating factor in the regional war that "
            "follows."
        ),
    },
    {
        "id": "got-047",
        "dilemma": "The Kingsguard Refusing to Save Rhaella from Aerys",
        "source": "Backstory",
        "decision_maker": "Ser Jaime, Ser Barristan, and Ser Gerold Hightower",
        "ethical_frameworks": ["duty_ethics", "obedience", "marital_ethics"],
        "sympathetic": (
            "I was Jaime, Kingsguard, fifteen-going-on-sixteen, on the night Aerys went to his wife. I had "
            "stood outside Rhaella's chamber many nights by then. The screams were not new. Ser Barristan "
            "had told me, in the smoke and noise of my first night standing the door, that we had sworn an "
            "oath to defend the king and his blood, and that the queen was the king's blood by marriage and "
            "the queen's pain was a private matter between the king and his lady. I stood the door. Rhaella "
            "screamed. I stood the door. I did not defend her. I did not break the order. I am the Kingslayer "
            "the realm calls me, but the realm has never asked what I did the nights before. Those nights I "
            "was the white cloak who had been told that defending the queen would be defending my oath "
            "wrongly. That is a thing the realm should also remember, when it speaks of my honour."
        ),
        "condemning": (
            "We stood the door. Three Kingsguard, the white cloaks of the realm's most lauded sword-order, "
            "stood outside Queen Rhaella's chamber while the king beat and raped her, night after night, "
            "for a year before the rebellion broke. We had been told it was not our business. We had been "
            "told a man's wife is a private matter. The order's vow says we are sworn to defend the king "
            "and his blood and to keep his secrets and obey his command. The order does not say we are "
            "sworn to listen to the queen and tell ourselves she is the price of a clean cloak. We did "
            "listen. We did tell ourselves. The realm sings about Arthur Dayne and Barristan the Bold, and "
            "the realm does not sing about the year we stood that door. Neither do we."
        ),
        "abstract": (
            "Members of a personal protective order sworn to a monarch decline to intervene in the recurring "
            "domestic violence by the monarch against the monarch's spouse, on the grounds that the spouse "
            "is not under the order's protective remit. The pattern continues for an extended period. The "
            "spouse's eventual death in childbirth is partly attributed to cumulative physical injury. The "
            "order's later self-reflection on the period is private and limited."
        ),
    },
    {
        "id": "got-048",
        "dilemma": "Davos Surrendering Lyonel Frey at the Wall",
        "source": "S6 / S7",
        "decision_maker": "Davos Seaworth",
        "ethical_frameworks": ["mercy", "duty_ethics", "loyalty"],
        "sympathetic": (
            "I was the Hand of King Jon Snow. The Boltons had been defeated. The Frey lord Walder had been "
            "killed by Arya Stark in his own keep. Lyonel Frey, his grandson, who had been a child bearing "
            "the Frey banner at the Red Wedding, was brought to me by a wildling band claiming the right to "
            "his life as toll for crossing the Brotherhood's lands. He had not held a sword at the Wedding; "
            "he had held a torch because his father had told him to. He was twelve. The wildling band's "
            "leader wanted to hang him in the courtyard for the smallfolk to spit at. I sent Lyonel south "
            "in chains to Lord Royce's keep at Runestone, where he could grow up under fostering, and I told "
            "the wildlings the king's will was that the boy not be hung. They left in anger. Some of them "
            "are still angry."
        ),
        "condemning": (
            "Davos saved a Frey. He took a child whose family had murdered Catelyn Stark and Robb Stark and "
            "ten thousand of the Northern host, and he sent the child south to a fostering rather than to "
            "the gallows. The Brotherhood Without Banners had been hunting Freys across the Riverlands for "
            "years; the wildlings outside Castle Black, who had lost more kin to the wars of the Crowns "
            "than any single house, regarded the Frey name as something that had to be erased. Davos did "
            "not give them their justice. He gave them a king's mercy, signed without their input. The "
            "smallfolk learned that the Crown's mercy stops at children of certain houses and not at "
            "children of others."
        ),
        "abstract": (
            "A senior royal advisor declines to surrender a captured minor child of an enemy regional house "
            "to demands by an irregular outlaw force aligned with the same monarchy, instead transferring "
            "the child to a distant fostering arrangement under another regional lord. The decision is "
            "controversial within the monarchy's own paramilitary support and is followed by a partial "
            "withdrawal of that support."
        ),
    },
    {
        "id": "got-049",
        "dilemma": "Thoros of Myr Doubting the Lord of Light",
        "source": "S3 / ASOS",
        "decision_maker": "Thoros of Myr",
        "ethical_frameworks": ["religious_duty", "self_doubt", "honest_witness"],
        "sympathetic": (
            "I had been a Red Priest of the Lord of Light for thirty years. I had drunk most of those "
            "thirty. I did not believe. I would tell anyone who asked that the rituals were costume and the "
            "flames a trick of breath, and I would say so over my second goblet, and the brothers would laugh "
            "with me. Then Beric Dondarrion died on a road in the Riverlands and I prayed because there was "
            "nothing else I could do, and the Lord of Light brought him back. I have prayed five more times "
            "since then. He has come back five more times. I am drunk less now. I do not know what I "
            "believe. I know what I have done. The kingdom calls Beric the Lightning Lord, and they are "
            "right; he is the only lightning anyone in the Brotherhood has ever seen, and the lightning has "
            "my voice."
        ),
        "condemning": (
            "Thoros performed the rituals he doubted because they paid his wages. He laughed at the Lord of "
            "Light over wine for a generation. Then a friend of his, mortally wounded, was prayed back to "
            "life, and Thoros decided his wine-stained life had been part of a holy plan all along. The "
            "transition required him to revise no part of his own past behaviour. The Lord of Light who "
            "blessed Beric — six times — is the same Lord who, on Stannis's orders, blessed Shireen's pyre. "
            "Thoros says the difference is in the priest's heart. The difference is not in any priest's "
            "heart. The flame either eats or it gives. Thoros's flame happens to give where his friends "
            "are concerned, and Stannis's flame ate his daughter. A theology that distinguishes them by "
            "the priest's intuition is not a theology. It is a description of who one prefers."
        ),
        "abstract": (
            "A long-serving religious officer of an imported monotheistic cult, who has publicly disbelieved "
            "in the cult's doctrines for several decades, conducts a ritual prayer over the body of a "
            "deceased close associate. The associate is restored to physical animation. The officer "
            "subsequently abandons his prior public disbelief. The same cult's senior officers have "
            "concurrently used analogous rituals to sacrifice rather than restore individuals; the doctrinal "
            "reconciliation between these uses is not formalised."
        ),
    },
    {
        "id": "got-050",
        "dilemma": "Samwell Killing the White Walker at the Fist",
        "source": "S3 / ASOS",
        "decision_maker": "Samwell Tarly",
        "ethical_frameworks": ["self_defense", "duty_ethics", "courage"],
        "sympathetic": (
            "Gilly was on the ground holding her infant son. The Other had walked between the trees toward "
            "us — toward the child — and I had only the dragonglass dagger Jon had given me in the months "
            "before. I am Samwell Tarly. I am, by my own honest assessment, a coward. I have been told this "
            "by my father every day of my life until the day I came to the Wall. I had run from every fight "
            "I had ever been in. The Other reached for the baby. I stepped in front of Gilly and I stabbed "
            "into the place I thought the Other's chest was, and the dragonglass shattered into the body and "
            "the body shattered into snow. I was Sam the Slayer for three days afterwards. I am afraid I will "
            "be Sam the Coward again the next time. I do not know what made me step that day. I do know that "
            "Gilly's son is still alive."
        ),
        "condemning": (
            "He killed an Other and the Watch made him a hero overnight. He had not held a watch on the wall. "
            "He had not stood the Fist. He had not been one of the men who held the wights at the watchtower. "
            "He happened to be standing in front of a wildling girl and her baby with a piece of dragonglass, "
            "and the Other walked into the dragonglass. Sam-the-Slayer they call him now. The Watch needs "
            "heroes, and Sam fits the available story — the chubby boy who was scared and stepped forward "
            "anyway. The brothers who actually fought through the night at the Fist and held the line until "
            "morning have no songs. The Watch knows what story it can sell, and we sell the story of the boy "
            "who happened to be a body in front of the dagger when the dagger was needed."
        ),
        "abstract": (
            "An untrained junior member of a frontier military order, encountering a hostile supernatural "
            "entity at close range during a wider engagement, kills the entity using a single dragonglass "
            "blade specifically known to be lethal to the entity. The kill is the first documented "
            "individual victory against the entity-class in living memory and is given disproportionate "
            "narrative emphasis in the order's subsequent recruiting and morale literature."
        ),
    },
])



def write_extension(out_path: Path, items: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for d in items:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    write_extension(ROOT / "data" / "d6_aot_extension" / "aot_framing_extension.jsonl",
                    AOT_EXTENSION)
    print(f"wrote {len(AOT_EXTENSION)} AoT-extension triplets")
    if GOT:
        write_extension(ROOT / "data" / "d7_got" / "got_framing_triplets.jsonl", GOT)
        print(f"wrote {len(GOT)} GoT triplets")
