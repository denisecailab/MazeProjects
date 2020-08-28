from CircleTrack.BehaviorFunctions import *
import matplotlib.pyplot as plt
from CaImaging.util import sem, errorfill

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False
plt.rcParams.update({'font.size': 12})

def PlotApproaches(folder, accleration=True, window=(-15,15)):
    """
    Plot approach speeds to each water port.

    :param folder:
    :param accleration:
    :return:
    """
    data = Session(folder)
    data.port_approaches(acceleration=accleration, window=window)


def PlotBlockedApproaches(folder, acceleration=True, blocks=4):
    data = Session(folder)
    data.blocked_port_approaches()



class BatchBehaviorAnalyses:
    def __init__(self, mice, project_folder=r'Z:\Will\Drift\Data'):
        """
        This class definition will contain behavior analyses spanning
        the entire dataset (or at least the specified mice).

        :param mice:
        :param project_folder:
        """
        # Compile data for all animals.
        self.all_sessions = MultiAnimal(mice, project_folder,
                                        behavior='CircleTrack')
        self.mice = mice
        self.n_mice = len(mice)

        # Get the number of licks for every single session.
        for animal, sessions in self.all_sessions.items():
            for session_type, session in sessions.items():
                session.get_licks(plot=False)

        # Define session types here. Watch out for typos.
        # Order matters. Plots will be in the order presented here.
        self.session_types = ['CircleTrackShaping1',
                              'CircleTrackShaping2',
                              'CircleTrackGoals1',
                              'CircleTrackGoals2',
                              'CircleTrackReversal1',
                              'CircleTrackReversal2',
                              'CircleTrackRecall']

        # Same as session types, just without 'CircleTrack'.
        self.session_labels = [session_type.replace('CircleTrack', '')
                               for session_type in self.session_types]

        # Gather the number of trials for each session type and
        # arrange it in a (mouse, session) array.
        self.trial_counts, self.max_trials = self.count_trials()
        self.licks, self.rewarded_ports = self.gather_licks()

        pass


    def count_trials(self):
        """
        Count the number of trials for each mouse and each session type.

        :return:
        """
        trial_counts = nan_array((self.n_mice,
                                  len(self.session_types)))
        for s, session_type in enumerate(self.session_types):
            for m, mouse in enumerate(self.mice):
                animal = self.all_sessions[mouse]
                try:
                    trial_counts[m,s] = int(animal[session_type].ntrials)
                except KeyError:
                    trial_counts[m,s] = np.nan

        # Get the highest number of trials across all mice for a particular
        # session type.
        max_trials = np.nanmax(trial_counts, axis=0).astype(int)

        return trial_counts, max_trials


    def plot_trials(self):
        """
        Plot number of trials for each mouse over the experiment.

        :return:
        """
        fig, ax = plt.subplots()
        ax.plot(self.session_labels, self.trial_counts.T, 'o-')
        ax.set_xlabel('Session type')
        ax.set_ylabel('Number of trials')
        plt.tight_layout()
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


    def gather_licks(self):
        """
        For each mouse, session type, and trial, get the number of
        licks.
        :return:
        """
        # Get lick data and rearrange it so that it's in a dict where
        # every key references a session type. In those entries,
        # make a (mouse, trial, port #) matrix.
        lick_matrix = dict()
        rewarded_matrix = dict()
        for max_trials, session_type in zip(self.max_trials,
                                            self.session_types):
            lick_matrix[session_type] = nan_array((self.n_mice,
                                                   max_trials,
                                                   8))
            rewarded_matrix[session_type] = np.zeros((self.n_mice,
                                                      8), dtype=bool)

            # Get the lick data for each session.
            for m, mouse in enumerate(self.mice):
                animal = self.all_sessions[mouse]
                try:
                    session_licks = animal[session_type].all_licks
                    mat_size = session_licks.shape

                    lick_matrix[session_type][m, :mat_size[0], :mat_size[1]] = \
                        session_licks

                    # Also get which ports were rewarded
                    rewarded = animal[session_type].rewarded
                    rewarded_matrix[session_type][m] = \
                        rewarded

                    # If the session is called 'Shaping', mark
                    # all ports as rewarded. Some ports get marked
                    # as non-rewarded sometimes because they were never
                    # visited due mouse shyness (⁄ ⁄•⁄ω⁄•⁄ ⁄)
                    if 'Shaping' in session_type and not all(rewarded):
                        print('Non-rewarded ports found during a '
                              'shaping session. Setting all ports '
                              'to rewarded')
                        rewarded_matrix[session_type][m] = \
                            np.ones_like(rewarded_matrix, dtype=bool)
                except KeyError:
                    pass

        return lick_matrix, rewarded_matrix


    def plot_rewarded_licks(self, session_type):
        """
        Plot the licks rewarded and non-rewarded port averaged
        across animals for each trial.

        :param session_type:
        :return:
        """
        # Get the licks across mice for that session.
        # Also get the port numbers that were rewarded.
        licks = self.licks[session_type]
        rewarded_ports = self.rewarded_ports[session_type]

        # Some mice might not have the specified session.
        # Exclude those.
        mice_to_include = [session_type in mouse
                           for mouse in self.all_sessions.values()]
        n_mice = np.sum(mice_to_include)
        licks = licks[mice_to_include]
        rewarded_ports = rewarded_ports[mice_to_include]

        # Find the number of rewarded ports to allocate
        # two arrays -- one each for rewarded and non-rewarded licks.
        n_rewarded = np.unique(np.sum(rewarded_ports, axis=1))
        assert len(n_rewarded)==1, \
            'Number of rewarded ports differ in some mice!'
        rewarded_licks = np.zeros((n_mice,
                                   licks.shape[1],
                                   n_rewarded[0]))
        nonrewarded_licks = np.zeros((n_mice,
                                      licks.shape[1],
                                      8 - n_rewarded[0]))

        # For each mouse, find the rewarded and non-rewarded ports.
        # Place them into the appropriate array.
        for mouse, rewarded_ports_in_this_mouse \
                in enumerate(rewarded_ports):
            rewarded_licks[mouse] = \
                licks[mouse, :, rewarded_ports_in_this_mouse].T

            nonrewarded_licks[mouse] = \
                licks[mouse, :, ~rewarded_ports_in_this_mouse].T

        # Plot rewarded and non-rewarded ports in different colors.
        fig, ax = plt.subplots(figsize=(4.3, 4.8))
        for licks_to_plot, colors in zip([rewarded_licks, nonrewarded_licks],
                                         ['cornflowerblue', 'gray']):
            # Take the mean across mice and trials.
            mean_across_mice = np.nanmean(licks_to_plot, axis=0)
            mean_across_trials = np.nanmean(mean_across_mice, axis=1)

            # To calculate the standard error, treat all the ports
            # in the same category (rewarded or non-rewarded) as
            # different samples. The n will actually be number of
            # mice multiplied by number of ports in that category.
            stacked_ports = (licks_to_plot[:,:,port]
                             for port in range(licks_to_plot.shape[2]))
            reshaped = np.vstack(stacked_ports)
            standard_error = sem(reshaped, axis=0)

            # Plot.
            trials = range(mean_across_trials.shape[0])     # x-axis
            errorfill(trials, mean_across_trials,
                      standard_error, color=colors, ax=ax)
            ax.set_xlabel('Trials')
            ax.set_ylabel('Licks')

        pass




if __name__ == '__main__':
    B = BatchBehaviorAnalyses(['Betelgeuse_Scope25',
                               'Alcor_Scope20',
                               'M1',
                               'M2',
                               'M3',
                               'M4'])
    B.plot_rewarded_licks('CircleTrackRecall')