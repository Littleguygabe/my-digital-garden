For the last few hours now I've been researching about potentially more effective methods on predicting volatility to be used for the Montecarlo Simulation during which I've effectively come across 2 main ways of generating volatility: Statically (my current approach) and dynamically (through the use of an MSGARCH model rather than just the current MS model)

### Fixed Volatility
With my current approach using the python package `statsmodels`' markov regression attribute which can produce a given number of regimes along with all the parameters needed for the monte carlo simulation. This means after fitting each regime will have its own parameter values which can then be extracted and used for the Montecarlo Simulation however even though this method provides us with a different volatility and mean of the dependent variable for the specific regime (in this case it's the day-on-day return of the stock) just because the stock may be in a given regime it doesn't guarantee that the volatility of the stock is going to be a constant value. This is where the next approach gets introduced.

### Variable Volatility Within Regimes Using an MSGARCH Model
This approach becomes a lot more complex as there are no mainstream python modules that can allow you to implement this model so this would mean building the majority of the workflow myself. 

The way this is implemented is by allowing the GARCH model and the Markov Switching (MS) identification program to communicate with each other during the fitting process which allows the 2 systems to contextualise the data between one another and adapt their metrics. 

##### An analogy:
*Think of the whole system like a sound engineer that needs to produce a song, if the engineer makes a really good vocal track then stops and separately records the drums for the song, although both the vocals and the drums sound great by themselves. **But** because both were made exclusively they are could clash really badly and sounds horrible together.* 

*This is why the sound engineer needs to be able to hear the drums when producing the vocals and the vocals producing the drums, so that when production is finished it all comes together nicely to produce a complete song.*

*In this case the vocals are the MS model and the drums are the GARCH model, even though both may work effectively by themselves when combined they may produce contradictory outputs and hence the pipeline won't produce meaningful results.*

So as the *sound engineer* I need to build a program that can tune both the MS model and the GARCH model at the same time so they are contextualised to one-another which will then improve the quality of the output rather than damaging it. Once this step is then complete rather than using the volatility defined by the MS model the program would take the parameters of the GARCH model associated with the current regime and then use these values to calculate the volatility during the running of the Montecarlo simulation. 

My other option at the moment is to pull the plug on coding this system in python and make the jump over to Rust as there already exists rust modules that facilitate exactly this type of pipeline - specifically the `MSGARCH` module - but the downside of this is that I would need to learn rust before being able to work on this project which could take 6 or so months and therefore I need to decide if I'm better spending the time learning a new programming language or if I want to spend the time researching about how to produce my own MSGARCH module - although this is typically seen as a PhD level of research.