
def full_identify(mat_file, db_file, svr_file):
    events = sp.read_mat(mat_file)
    events = sp.filter_by_time(events, 0.5, 20)
    sp.normalize(events)
    peptide = events[0].peptide
    num_peaks = len(peptide) + 3
    nano_hmm = NanoHMM(len(peptide), svr_file)

    database = {}
    target_id = None
    for seq in SeqIO.parse(db_file, "fasta"):
        database[seq.id] = str(seq.seq)[:len(peptide)]
        if database[seq.id] == peptide:
            target_id = seq.id

    boxes = []
    for avg in xrange(1, 21):
        p_values = []
        for _ in xrange(avg):
            clusters = sp.get_averages(events, avg)
            for cluster in clusters:
                rankings = rank_db_proteins(nano_hmm, cluster.consensus, database)
                target_rank = None
                for i, prot in enumerate(rankings):
                    if prot[0] == target_id:
                        target_rank = i
                        break

                #p_value = nano_hmm.compute_pvalue_raw(discr_signal, peptide)
                p_value = float(target_rank) / len(database)
                p_values.append(p_value)

        boxes.append(p_values)
        print(avg, np.median(p_values))

    for b in boxes:
        print(",".join(map(str, b)))



