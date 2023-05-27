from discord.ext import commands
import discord
import yaml
import random

from core.classes import Cog_Extension
import core.function as function

def load_sentences():
    global sentences
    with open('data/sentences.yml', encoding='utf8') as file:
        sentences = yaml.load(file, Loader=yaml.CLoader)

class Tools(Cog_Extension):
    @commands.command()
    async def ping(self, ctx: commands.Context):
        await ctx.send(f'ping: {self.bot.latency * 1000} (ms)')
        
async def setup(bot):
    await bot.add_cog(Tools(bot))