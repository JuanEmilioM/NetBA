from nba_api.stats.static import teams
from nba_api.stats.endpoints import (LeagueGameLog, TeamGameLogs,
                                     LeagueDashTeamStats, BoxScoreAdvancedV2)
import numpy as np
import pandas as pd

features = ['GAME_ID','WL','TEAM_ID','MIN','PTS','FG_PCT','FG3_PCT','FT_PCT','DREB','OREB','AST',
           'TOV','STL','BLK','PLUS_MINUS']

features_advanced = ['GAME_ID', 'TEAM_ID', 'OFF_RATING', 'DEF_RATING','NET_RATING',
                     'AST_PCT', 'AST_TOV', 'AST_RATIO',
                     'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT','PACE', 'POSS', 'PIE']

name_teams = ['Atlanta Hawks',
 'Boston Celtics',
 'Cleveland Cavaliers',
 'New Orleans Pelicans',
 'Chicago Bulls',
 'Dallas Mavericks',
 'Denver Nuggets',
 'Golden State Warriors',
 'Houston Rockets',
 'Los Angeles Clippers',
 'Los Angeles Lakers',
 'Miami Heat',
 'Milwaukee Bucks',
 'Minnesota Timberwolves',
 'Brooklyn Nets',
 'New York Knicks',
 'Orlando Magic',
 'Indiana Pacers',
 'Philadelphia 76ers',
 'Phoenix Suns',
 'Portland Trail Blazers',
 'Sacramento Kings',
 'San Antonio Spurs',
 'Oklahoma City Thunder',
 'Toronto Raptors',
 'Utah Jazz',
 'Memphis Grizzlies',
 'Washington Wizards',
 'Detroit Pistons',
 'Charlotte Hornets']

abv_teams = ['ATL', 'BOS', 'CLE', 'NOP', 'CHI', 'DAL', 'DEN', 'GSW', 'HOU', 'LAC', 'LAL', 'MIA', 'MIL',
 'MIN','BKN', 'NYK', 'ORL', 'IND', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'OKC', 'TOR', 'UTA', 'MEM',
 'WAS', 'DET', 'CHA']

id_teams = [1610612737, 1610612738, 1610612739, 1610612740, 1610612741, 1610612742, 1610612743,
 1610612744, 1610612745, 1610612746, 1610612747, 1610612748, 1610612749, 1610612750, 1610612751,
 1610612752, 1610612753, 1610612754, 1610612755, 1610612756, 1610612757, 1610612758, 1610612759,
 1610612760, 1610612761, 1610612762, 1610612763, 1610612764, 1610612765, 1610612766]

season_start_date = '10/01' # typical start date of NBA regular season

class Loader:
    """ Atributos
        stats_teams. Final stats of each team.
        games_teams_prom. Basic stats for each team from each game.
        games_advanced. Advanced stats for each team from each game.
    """
    def __init__(self, season, name, date_start='', date_end='',
                 advanced_stats=False, proxy='', season_type='Regular Season'):
        """

        Parameters
        ----------
        season : string
            season in the format year start-year end, e.g. '2015-16'.
        name : string
            name for the file.
        date_start : string, optional
            usage: MM/DD/YYY. The default is ''.
        date_end : string, optional
            usage: MM/DD/YYY. The default is ''.
        advanced_stats: bool
            whether or not obtain advanced stats. The default is False.
        proxy : string
            proxy to be used. The default is ''.
        season_type : string
            Regular season or playoffs. The default is Regular Season.

        Returns
        -------
        an instance of Loader.

        """
        if (season_type != 'Regular Season') and (season_type != 'Playoffs'):
            raise RuntimeError('Invalid season type')

        if date_start == '':
            date_start = season_start_date + '/' + season[:4]

        self.stats_teams = []
        self.games_teams_prom = []
        self.games_advanced = []

        # stats teams
        for i in id_teams:
            x = LeagueDashTeamStats(team_id_nullable=i, season=season,
                date_from_nullable=date_start, date_to_nullable=date_end, proxy=proxy).get_data_frames()[0]
            x = x[features[2:]] # selection of stats of interest

            # normalization of stats by played minutes
            x[features[8:]] /= x['MIN'][0]
            x['PTS'] /= x['MIN'][0]
            self.stats_teams.append(x)
        self.stats_teams = pd.concat(self.stats_teams)

        # stats teams prom
        for i in id_teams:
            x = TeamGameLogs(date_from_nullable=date_start, team_id_nullable=i,
              season_nullable=season, proxy=proxy, season_type_nullable=season_type).get_data_frames()[0]
            x = x.sort_values(by=['GAME_ID'])
            x = x[features].to_numpy()
            y = np.ndarray(shape=(x.shape[0],len(features)), dtype='object')
            # normalization of stats by played minutes
            y = x
            y[:, 8:] /= y[:, 3].reshape((x.shape[0],1))
            y[:, 4] /= y[:, 3]

            for j in range(x.shape[0]):
                y[j,:4] = x[j, :4]
                y[j, 4:] = x[:j+1, 4:].mean(axis=0)

            self.games_teams_prom.append(pd.DataFrame(data=y, columns=features))

        # data accommodation to save it to a .csv file
        games = pd.concat(self.games_teams_prom).sort_values(by=['GAME_ID'])

        WL = games['WL']
        WL = WL.replace(to_replace={'W':1, 'L':0}) # 1 if team 1 WIN. 0 if team 1 LOSS
        WL = WL.to_numpy()

        columns = features[4:]
        columns = np.hstack((columns, columns, np.array('WL')))

        N = int(games.shape[0] / 2)

        team1 = []
        team2 = []
        WL_team1 = []
        #WL_team2 = []
        games = games[features[4:]].to_numpy()
        i = 0
        while i<N:
            team1.append(games[2*i,:])
            team2.append(games[2*i+1,:])
            WL_team1.append(WL[2*i+1])
            #WL_team2.append(WL[2*i+1])
            i+=1

        team1 = np.array(team1)
        team2 = np.array(team2)
        WL_team1 = np.array(WL_team1).reshape((N,1))
        #WL_team2 = np.array(WL_team2).reshape((N,1))
        #data = pd.DataFrame(np.hstack((team1, team2, WL_team1, WL_team2)), columns=columns)
        data = pd.DataFrame(np.hstack((team1, team2, WL_team1)), columns=columns)
        data.to_csv('files/' + name + '.csv', index=False)

        # load and save of advanced stats
        if advanced_stats:
            for i in range(len(id_teams)): # iterator over teams
                ids_games = self.games_teams_prom[i]['GAME_ID']
                x = []

                for id_ in ids_games: # iterator over each game
                    y = BoxScoreAdvancedV2(game_id = id_, proxy=proxy).team_stats.get_data_frame()
                    y = y.loc[y['TEAM_ID'] == id_teams[i]]
                    y = y[features_advanced]
                    x.append(y)
                print(i)
                self.games_advanced.append(pd.concat(x))

            games = []
            for i in range(len(id_teams)): # iterator over teams
                x = self.games_advanced[i].to_numpy()
                y = np.ndarray(shape=(x.shape[0],14), dtype='object')
                minutes = self.games_teams_prom[i]['MIN'].to_numpy().reshape((x.shape[0],1))

                for j in range(x.shape[0]): # iterator over games
                    y[j,:2] = x[j,:2]
                    y[j,2:] = x[:j+1,2:].mean(axis=0)

                # normalization of stats by played minutes
                y[:, 2:] /= minutes
                games.append(pd.DataFrame(y, columns=features_advanced))

            # self.games_advanced = []
            self.games_advanced = games

            # data accommodation to save it to a .csv file
            games = pd.concat(self.games_advanced).sort_values(by=['GAME_ID'])
            WL = pd.concat(self.games_teams_prom)['WL']
            WL = WL.replace(to_replace={'W':1, 'L':0}) # 1 if team 1 WIN. 0 if team 1 LOSS
            WL = WL.to_numpy()

            columns = features_advanced[2:]
            columns = np.hstack((columns, columns, np.array('WL')))

            N = int(games.shape[0] / 2)

            team1 = []
            team2 = []
            WL_team1 = []
            games = games[features_advanced[2:]].to_numpy()
            i = 0
            while i<N:
                team1.append(games[2*i,:])
                team2.append(games[2*i+1,:])
                WL_team1.append(WL[2*i])
                i+=1

            team1 = np.array(team1)
            team2 = np.array(team2)
            WL_team1 = np.array(WL_team1).reshape((N,1))
            data = pd.DataFrame(np.hstack((team1, team2, WL_team1)), columns=columns)
            data.to_csv('files/' + name + '-advanced.csv', index=False)      
