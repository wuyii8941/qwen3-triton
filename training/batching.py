"""Fixed-size microbatch grouping; optimizer step counts complete groups only."""
def accumulation_groups(loader, accumulation):
    if accumulation<1:raise ValueError('Positive accumulation required')
    group=[]
    for batch in loader:
        group.append(batch)
        if len(group)==accumulation:
            yield tuple(group)
            group=[]
    # Incomplete tails are not used for an update. Resume offset counts only
    # completed optimizer steps, so the next loader starts at the unused data.
