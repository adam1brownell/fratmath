import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
import copy

import gspread
from oauth2client.service_account import ServiceAccountCredentials

import sys

import plotly.graph_objects as go
import plotly.figure_factory as ff

def pullWorkbook(gsheet_name, gsheet_workbook):

	"""
	Pulls data from Google Sheets workbook.
	Requires GSheet API and Secrets (secret_key.json)

	THIS WON'T RUN WITHOUT YOUR OWN SECRETS
	"""
    
    scope =[ "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive"
       ]

    creds = ServiceAccountCredentials.from_json_keyfile_name("secret_key.json", scopes=scope)

    file = gspread.authorize(creds)

    workbook = file.open(gsheet_workbook)

    sheet = workbook.worksheet(gsheet_name)
    
    dataframe = pd.DataFrame(sheet.get_all_records())
    
    return(dataframe)



def ssim(games_dict):
    """
    Randomly select players in a game to simulate
    1/2 vs 3/4
    
    Input:
    - games_dict: dictionary of aggregate tourney values
        - {'player':[ELO, wins, loses]...}
    
    Output:
    - elo_sim_dict: dictionary of simulated ELOs
     - {'player':[EL0_1, ELO_2...]}
    
    Determining Winner:
     - If any of them only have wins/loses left, that team wins/loses
     - Redraw if there are contradictions 
         - teammates cannot both either win or lose
         - teams cannot resolve a win/lose decision
     - Otherwise, team that has higher culmative win perc is winner
     
     Update games_dict with
      - updated ELO
      - updated win/lose
      
      Exit if < 4 players remaining 
      
      Is there a more efficient DP solution? Probably but I'm lazy.
    """
    
    K = 30

    elo_sim_dict = dict()
    
#     print("KEYS:",len(list(games_dict.keys())),list(games_dict.keys()))

    for i in range(1000):

        # Update probabilities of being drawn based on games left
        s = 0
        player_probs = []
        for p in games_dict.keys():
        
            games_left = games_dict[p][1]+games_dict[p][2]
            player_probs.append(games_left)
            s += games_left
            
        player_probs =  [z/s for z in player_probs]
        
        # Draw players
        try:
            players = np.random.choice(list(games_dict.keys()),4, p=player_probs, replace=False)
        except:
            print(games_dict)
            print("*****")
            print("*****")
            for p in games_dict.keys():
                print(p)
                print("Wins:",games_dict[p][1])
                print("Losses:",games_dict[p][2])
                print("Remaining:",games_dict[p][1]+games_dict[p][2])
                print("Exit:",games_dict[p][1]+games_dict[p][2] == 0)
            print(sdfgsdfgs)

        p1 = games_dict[players[0]]
        p2 = games_dict[players[1]]
        p3 = games_dict[players[2]]
        p4 = games_dict[players[3]]

        team_win = None

        one_must_win = False
        one_must_lose = False
        two_must_win = False
        two_must_lose = False


        # Check ending cases
        if p1[2] == 0 or p2[2] == 0:
            one_must_win = True
        if p1[1] == 0 or p2[1] == 0:
            one_must_lose = True

        if p3[2] == 0 or p4[2] == 0:
            two_must_win = True
        if p3[1] == 0 or p4[1] == 0:
            two_must_lose = True 

        # Team Contradiction
        if (one_must_win & one_must_lose) or (two_must_win & two_must_lose):
            continue

        # Game Contradiction
        elif (two_must_win & one_must_win) or (two_must_lose & one_must_lose):
            continue


        # Calc win percentage
        one_wins = p1[1]+p2[1]
        one_loses = p1[2]+p2[2]
        one_perc = one_wins/(one_wins+one_loses)

        two_wins = p3[1]+p4[1]
        two_loses = p3[2]+p4[2]
        two_perc = two_wins/(two_wins+two_loses)

        if one_perc > two_perc:
            team_win = "Team 1"
        else:
            team_win = "Team 2"


        # Team ELOs
        oneELO = p1[0] + p2[0]
        twoELO = p3[0] + p4[0]

        # Expected Win %
        E_one = 1/(1+10**((twoELO-oneELO)/400))

        E_two = 1/(1+10**((oneELO-twoELO)/400))

        # Outcomes
        if team_win == "Team 1":
            S_one = 1
            S_two = 0

            p1[1] -= 1
            p2[1] -= 1
            p3[2] -= 1
            p4[2] -= 1

        elif team_win == "Team 2":
            S_one = 0
            S_two = 1

            p1[2] -= 1
            p2[2] -= 1
            p3[1] -= 1
            p4[1] -= 1
        else:
            raise Exception("Mispelled Team Win, you idiot")

        # Update ELO
        U_one = round(K * (S_one-E_one))
        U_two = round(K * (S_two-E_two))

        p1[0] += round(U_one/2)
        p2[0] += round(U_one/2)
        p3[0] += round(U_two/2)
        p4[0] += round(U_two/2)
        
        for p in players:
            if games_dict[p][1]+games_dict[p][2] == 0:
                elo_sim_dict[p] = games_dict[p][0]
                games_dict.pop(p, None)

        if len(games_dict.keys())< 4:
            break
    
    for p in games_dict.keys():
        elo_sim_dict[p] =  games_dict[p][0]
        
    return(elo_sim_dict,games_dict)

def aggSimELO(elo_dict, gsheet_name, gsheet_workbook="Super Bowl Corners"):
    """
    This function simulates ELO ratings from aggregate Corners results
    Input:
     - elo_dict: The ELO scores of members entering SB Weekend
     - gsheet_name:
     - gsheet_workbook: The workbook of corners data. Default is Super Bowl Corners'
     
     Output:
     - new_elo_dict: The ELO scores of members leaving SB Weekend
         - This can/should be added to history_elo_dict
    """
    
    dataframe = pullWorkbook(gsheet_name,gsheet_workbook)
    
    # Make sure all new players added. ELO starts at 1000
    for player in dataframe.Person.values:
        if player not in elo_dict.keys():
            elo_dict[player] = 1000
            
    # Generate games_dict for simulation
    pregames_dict = {}
    postgames_dict = {}
    for i,j in dataframe.iterrows():
        pregames_dict[j.Person] = [elo_dict[j.Person],j.Wins,j.Losses]
        postgames_dict[j.Person] = []
    
    
    
    # Gets to goodsim ~2.5% of the time
    # 100K sims will net 3K examples
    print(f"Simulating {gsheet_name}...")
    goodsim_cnt = 0
    
    for i in range(100):
        if i % 1000 == 0:
            print(f'run sim {i}...')
        elo_sim_dict,games_dict = ssim(copy.deepcopy(pregames_dict))

        goodsim = True
        for leftovers in games_dict.keys():
            p = games_dict[leftovers]

            # If there are more than 2 games left for any player, bad sim.
            if p[1]+p[2] > 2:
                goodsim = False

            # Why are there sometimes negatives?
            elif (p[1] < 0) or (p[2] < 0):
                goodsim = False

        if goodsim:
            goodsim_cnt += 1
#             print(f"{goodsim_cnt} out of {i+1} ({100*goodsim_cnt/(i+1):.2f}%)")
            for p in elo_sim_dict.keys():
                sim_elo = elo_sim_dict[p]

                if p in games_dict.keys():
                    sim_elo+= (games_dict[p][1]*8) # +8pts for a leftover win
                    sim_elo+= (games_dict[p][2]*8) # -8pts for a leftover lose

                postgames_dict[p].append(sim_elo)
                
        ## Add visuals here is you want
        
    new_elo_dict = dict()

    for player, elos in postgames_dict.items():
        new_elo_dict[player] = round(np.mean(elos))

    return(new_elo_dict)

def gamesELO(elo_dict, history_elo, gsheet_name, gsheet_workbook="Super Bowl Corners"):
    """
    This function simulates ELO ratings from game-level Corners results
    Input:
     - elo_dict: The ELO scores of players entering SB Weekend
     - history_elo: The ELO scores of players over time
     - gsheet_name:
     - gsheet_workbook: The workbook of corners data. Default is 'Super Bowl Corners'
     
     Output:
     - new_elo_dict: The ELO scores of members leaving SB Weekend
     - trip_history: game-level history of ELO scores
    """
    

    ## ISSUE: There is no workbook in Casey's pages that fits this format rn
    dataframe = pullWorkbook(gsheet_name,gsheet_workbook)

    new_elo_dict = copy.deepcopy(elo_dict)

    K = 30
    
    game_id = pd.to_numeric(history_elo['gameID'], errors='coerce') \
                                  .dropna().astype(int).values[-1]
    
    for i in range(len(dataframe)):

        row = dataframe.iloc[i]
        player1a = row['1A']
        player1b = row['1B']
        player2a = row['2A']
        player2b = row['2B'] 
        team_win = row['Win']
        
        players = [player1a, player1b, player2a, player2b]
        
        if player1a == "":
            continue
        
        # Add new players
        for player_name in players:
            if player_name not in new_elo_dict.keys():
                new_elo_dict[player_name] = 1000

                history_elo[player_name] = ""
                history_elo.loc[len(history_elo)-1,player_name] = 1000
        
        team1_str = f"{player1a} & {player1b}"
        team2_str = f"{player2a} & {player2b}"
                
        # Team ELOs
        oneELO = new_elo_dict[player1a] + new_elo_dict[player1b]
        twoELO = new_elo_dict[player2a] + new_elo_dict[player2b]

        # Expected Win %
        E_one = 1/(1+10**((twoELO-oneELO)/400))

        E_two = 1/(1+10**((oneELO-twoELO)/400))
    
        # Outcome
        if team_win == "Team 1":
            S_one = 1
            S_two = 0
        elif team_win == "Team 2":
            S_one = 0
            S_two = 1
        else:
            raise Exception("Mispelled Team Win, you idiot")

        # Update ELO
        U_one = round(K * (S_one-E_one))
        U_two = round(K * (S_two-E_two))

        new_elo_dict[player1a] += round(U_one/2)
        new_elo_dict[player1b] += round(U_one/2)
        new_elo_dict[player2a] += round(U_two/2)
        new_elo_dict[player2b] += round(U_two/2)
            
        gamedict = {"Team1":team1_str,"Team2":team2_str, 
                    "favored": "Team 1", "outcome":row['Win']}
        
        for player in players:
            gamedict[player] = new_elo_dict[player]
        
        diff = oneELO - twoELO
        if abs(diff) <= 10:
            gamedict['favored'] = "Toss-up"
        elif diff < 0:
            gamedict['favored'] = "Team 1"
            
        history_elo.loc[len(history_elo)] = gamedict
        history_elo.loc[len(history_elo)-1,'gameID'] = game_id+i+1
        
        
    return(new_elo_dict,history_elo)

def calculateELO(austin_redo=False, denver_redo=False, 
                 chicago_redo=False, miami_redo=False):
    """
    Calcuates ELO for Corners players over the past 5years
    
    Input:
     - Which year to rerun
     - Have hardcoded values for the past few years to save time on sim
     
    Output
    - current_elo: everyone's current ELO
    - history_elo: everyoe's ELO over time. For simmed/aggregate years
        this is a single number. For years we have game-level data
        this is after every match
        
    Visual?
    
    """
    
    def getCurrent(history_elo):
        """
        Pulls most recent plauer ELO from history ELO
        """
        elo_dict = {}
        skip_cols = ['gameID','Team1','Team2','favored','outcome']
        
        for col in history_elo.columns:
            if col in skip_cols:
                continue
            else:
                elo_dict[col] = pd.to_numeric(history_elo[col], errors='coerce') \
                                  .dropna().astype(int).iloc[-1]
        return(elo_dict)
    
    def addGameLevelCols(history_elo):
        glevel_cols = ['Team1','Team2','favored','outcome']
        glevel_cols.reverse()
        for col in glevel_cols:
            history_elo.insert(0,col,"")
            
        # Move column 'B' to the front
        column_to_move = 'gameID'
        column = history_elo.pop(column_to_move)
        history_elo.insert(0, column_to_move, column)
            
    
    def updateDF(elo_dict, history_elo,games):
        """
        Adds ELOs & new players to history_elo
        """
        
        # Add new players on last row
        for player in elo_dict.keys():
            if player not in history_elo.columns:
                history_elo[player] = ""
                history_elo.loc[len(history_elo)-1,player] = 1000
           
        # Add agg ELO and games played
        history_elo.loc[len(history_elo)] = elo_dict
        history_elo.loc[len(history_elo)-1,'gameID'] = pd.to_numeric(history_elo['gameID'], errors='coerce') \
                                                         .dropna().astype(int).iloc[-1]+games
        return
    
    history_elo = pd.DataFrame()
    history_elo['gameID'] = [0]
    
    # Austin
    if austin_redo:
        print("Rerunning Austin Simulations")
        austin_elo = aggSimELO({},"Austin 2019")
        print("\t",austin_elo)
    else:
        austin_elo = {  "Cat":1041,
                        "Bear":1042,
                        "Anteater":1046,
                        "Llama": 994,
                        "Viper":959,
                        "Elephant":955
        }
        
    # Add all players for first time. Austin specific
    # TODO: Make this work with updateDF()
    for player in austin_elo.keys():
        if player not in history_elo.columns:
            history_elo[player] = [1000]
            
    # Add agg ELO and games played
    history_elo.loc[len(history_elo)] = austin_elo
    history_elo.loc[len(history_elo)-1,'gameID'] = 32
    
    
    # Denver
    if denver_redo:
        print("Rerunning Denver Simulations")
        updated_elos = getCurrent(history_elo)
        denver_elo = aggSimELO(updated_elos,"Denver 2020")
        print("\t",denver_elo)
    else:
        denver_elo = {'Turtle': 1043, 
                      'Condor': 1043, 
                      'Viper': 1044, 
                      'Bear': 1044, 
                      'Cat': 1023, 
                      'Anteater': 1027, 
                      'Alligator': 1001, 
                      'Jaguar': 973, 
                      'Elephant': 944, 
                      'Collie': 972}
    
    updateDF(denver_elo, history_elo, 34)

    # Chicago
    if chicago_redo:
        print("Rerunning Chicago Simulations")
        updated_elos = getCurrent(history_elo)
        chicago_elo = aggSimELO(updated_elos,"Chicago 2022")
        print("\t",chicago_elo)
    else:
        chicago_elo = {'Cat': 1070, 
                       'Anteater': 1050, 
                       'Kangaroo': 1009, 
                       'Collie': 1003, 
                       'Drake': 991, 
                       'Viper': 960, 
                       'Rabbit': 974}
    
    updateDF(chicago_elo, history_elo, 34)
    
    # Miami
    print("Rerunning Miami")
    updated_elos = getCurrent(history_elo)
    addGameLevelCols(history_elo)
    miami_elo,history_elo = gamesELO(updated_elos, history_elo, "The Games", gsheet_workbook="Corners ELO")
    
    # End
    current_elo = dict(sorted(getCurrent(history_elo).items(), key=lambda item: item[1], reverse=True))
    history_elo = history_elo.fillna("")
    
    return(current_elo, history_elo)


if __name__ == "__main__":

	# RUN IT
	current_elo, history_elo = calculateELO(austin_redo=False, denver_redo=False, 
                                        chicago_redo=False)


	### This is all charting/visualizing code, and I was too lazy to move it to it's on
	### function. TODO
	charting_elo = copy.deepcopy(history_elo)
	# charting_elo.loc[len(charting_elo)] = current_elo

	layout = go.Layout(
	                title='Corners ELO',
	                paper_bgcolor='rgba(0,0,0,0)',
	                plot_bgcolor='rgba(0,0,0,0)',
	                xaxis = dict(title = 'Games'),
	                yaxis = dict(title = 'ELO'),
	)


	c = [
	    '#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a',
	    '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f',
	    '#cab2d6', 'grey', '#66c2a5', '#fc8d62', '#8da0cb',
	    '#e78ac3', '#a1dab4', '#e5c494', '#9e0142', '#d53e4f'
	]

	idx = 0


	fig = go.Figure(layout=layout)

	annotations = []

	num_players = len(current_elo.keys())
	y1 = 900
	y2 = 1100
	ydiff = (y2-y1)/num_players

	for player in current_elo.keys():
	    p = charting_elo[player]
	    
	    # find when they began playing
	    for val in p.values:
	        if val == 1000:
	            break
	        i += 1
	    try:
	        y = p.replace(to_replace="",value=np.nan).interpolate(method='polynomial', order=2)
	    except:
	        
	        # These players are mostly one-offs and so don't have enough data
	        y = p
	    x = charting_elo.gameID
	    
	    ## trace1 = line of all games played, no hover no text
	    trace1 = go.Scatter(x=x, y=y,name='', opacity=0.5, line={'shape': 'spline', 'smoothing': 1.3, 'color':c[idx]},
	                        legendgroup=player, hoverinfo='skip') 
	    
	    fig.add_trace(trace1)
	    
	    ## trace2 = hovermarkers for all games played
	    ##          a little bit busy IMO
	#     trace2 = go.Scatter(x=charting_elo.gameID,y=p,mode='markers',marker=dict(color=c[idx]), 
	#                         name=f"{player}:<br>{current_elo[player]}",legendgroup=player, 
	#                         hovertext=p, hoverinfo="text")
	    
	#     fig.add_trace(trace2)

	    ## trace3 = hovermarkers of start/current for each player
	    import numbers

	    x2 = charting_elo[['gameID',player]][[isinstance(i,numbers.Number) for i in charting_elo[player].values]]
	    lastg = max(x2.gameID)
	    lastg_score = x2[x2.gameID==lastg][player].values[0]
	    
	    trace3 = go.Scatter(x=[min(x2.gameID),lastg],y=[1000,lastg_score],
	                        mode='markers',marker=dict(color=c[idx],symbol='circle',size=8), 
	                        name=f"{player}:<br>{current_elo[player]}",legendgroup=player, 
	                        hovertext=f"{player}:<br>ELO: {int(lastg_score)}", hoverinfo="text")
	    
	    fig.add_trace(trace3)
	    
	    ## TODO: Add total games played...

	    fig.add_annotation(
	        x=max(charting_elo.gameID)+6,
	        y=y2-(ydiff*idx),
	        xref='x',
	        yref='y',
	        text=f"<b><span style='color:{c[idx]}'>{player}:</span> {current_elo[player]}</b>",
	        showarrow=False,
	        font=dict(size=7)  # Set the font size
	    )
	    
	    idx += 1
	    
	locations = ['Austin','Denver','Chicago','Miami']
	breaks = [0,32,66,100,134]
	for i in range(0,len(breaks)-1):
	    fig.add_annotation(
	        x=(breaks[i]+breaks[i+1])/2,
	        y=900,
	        xref='x',
	        yref='y',
	        text=locations[i],
	        showarrow=False,
	        font=dict(size=7)  # Set the font size
	    )
	    
	    if (i == 0):
	        continue
	    
	    fig.add_shape(
	        go.layout.Shape(
	            type='line',
	            x0=breaks[i],
	            x1=breaks[i],
	            y0=0,
	            y1=100,
	            xref='x',
	            yref='paper',
	            opacity=0.5,
	            line=dict(color='lightgrey', width=2, dash='dot')
	        )
	    )


	fig.show() 