import asyncio
import os
import discord
from discord.ext import commands
import core.function as function
from train_transformer import custom_standardization, custom_split


bot = commands.Bot(command_prefix='-', intents=discord.Intents.all())

async def main():
    @bot.event
    async def on_ready():
        function.print_detail(memo='INFO', obj='Bot is Ready')
        await bot.change_presence(status=discord.Status.online, activity=discord.Game(function.open_json('./data/config.json')['playinggame']))

    @bot.command()
    @commands.is_owner()
    async def load(ctx: commands.Context, extension):
        await bot.load_extension(f'cmds.{extension}')
        await ctx.send(f'Loaded {extension} successfully')
        function.print_detail(memo='INFO',user=ctx.author, guild=ctx.guild, channel=ctx.message.channel, obj=f'{extension}.py loaded successfully')


    @bot.command(aliases=['ul'])
    @commands.is_owner()
    async def unload(ctx: commands.Context, extension):
        await bot.unload_extension(f'cmds.{extension}')
        await ctx.send(f'Unloaded {extension} successfully')
        function.print_detail(memo='INFO',user=ctx.author, guild=ctx.guild, channel=ctx.message.channel, obj=f'{extension}.py unloaded successfully')


    @bot.command(aliases=['rl'])
    @commands.is_owner()
    async def reload(ctx: commands.Context, extension):
        await bot.reload_extension(f'cmds.{extension}')
        await ctx.send(f'Reloaded {extension} successfully')
        function.print_detail(memo='INFO', user=ctx.author, guild=ctx.guild, channel=ctx.message.channel, obj=f'{extension}.py reloaded successfully')

    async with bot:
        for file in os.listdir('./cmds'):
            if file.endswith('.py') and file != 'data.py':
                await bot.load_extension(f'cmds.{file[:-3]}')
                function.print_detail(memo='INFO', obj=f'{file} loaded successfully')
        await bot.start(function.open_json('config.json')['token']['discord'])


asyncio.run(main())