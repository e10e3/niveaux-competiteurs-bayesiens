namespace myApp

{
	using System;
	using System.Linq;
	using Microsoft.ML.Probabilistic;
	using Microsoft.ML.Probabilistic.Distributions;
	using Microsoft.ML.Probabilistic.Models;
	using Range = Microsoft.ML.Probabilistic.Models.Range;

	class Program
	{

		static void Main(string[] args)
		{
			// *********************************
			//     Création des données
			// *********************************
			// The winner and loser in each of 6 samples games
			var winnerData = new[] { 0, 2, 2, 1, 2, 0, 5, 4, 4, 6, 7, 6 };
			var loserData = new[] { 1, 3, 0, 3, 1, 3, 6, 7, 5, 4, 5, 7 };
			var drawData = new[] { 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };

			// Define the statistical model as a probabilistic program
			var game = new Range(winnerData.Length);
			var player = new Range(winnerData.Concat(loserData).Max() + 1);
			var playerSkills = Variable.Array<double>(player);
			playerSkills[player] = Variable.GaussianFromMeanAndVariance(6, 9).ForEach(player);

			var winners = Variable.Array<int>(game);
			var losers = Variable.Array<int>(game);
			var draws = Variable.Array<int>(game);

			// *******************************************
			//    Déclaration de la méthode d'inférence
			// *******************************************
			using (Variable.ForEach(game))
			{
				// The player performance is a noisy version of their skill
				Variable<double> winnerPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[winners[game]], 1.0);
				Variable<double> loserPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[losers[game]], 1.0);
				Variable<int> draw = draws[game];
				var drawMargin = Variable.New<double>().Named("drawMargin");
				var drawMarginPrior = Variable.New<Gaussian>().Named("drawMarginPrior");
				drawMarginPrior.SetTo(new Gaussian(1, 10));
				drawMargin.SetTo(Variable<double>.Random(drawMarginPrior));
				Variable.ConstrainTrue(drawMargin > 0);

				var diff = (winnerPerformance - loserPerformance).Named("diff");

				using (Variable.Case(draw, 0))
				{
					// The winner performed better in this game
					Variable.ConstrainTrue(winnerPerformance > loserPerformance);
				}

				using (Variable.Case(draw, 1))
				{
					// In a case of a draw, the difference is lower than the margin
					Variable.ConstrainBetween(diff, -drawMargin, drawMargin);
				}
			}

			// Attach the data to the model
			winners.ObservedValue = winnerData;
			losers.ObservedValue = loserData;
			draws.ObservedValue = drawData;


			// **********************************
			//    Effectuation de l'inférence
			// **********************************

			// Run inference
			var inferenceEngine = new InferenceEngine();
			var inferredSkills = inferenceEngine.Infer<Gaussian[]>(playerSkills);


			// **********************************
			//      Affichage
			// **********************************
			// The inferred skills are uncertain, which is captured in their variance
			var orderedPlayerSkills = inferredSkills
			.Select((s, i) => new { Player = i, Skill = s })
			.OrderByDescending(ps => ps.Skill.GetMean());

			foreach (var playerSkill in orderedPlayerSkills)
			{
				Console.WriteLine($"Player {playerSkill.Player} skill: {playerSkill.Skill}");
			}
		}
	}
}
