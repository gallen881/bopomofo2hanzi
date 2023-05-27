from discord.ext import commands
import discord

from core.classes import Cog_Extension
import core.function as function
import func
import translator

class Events(Cog_Extension):
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        message.content = func.end_fix(message.content.lower())
        if func.IsZhInputs(message.content):
            function.print_detail('INFO', message.author, message.guild, message.channel, f'Get message: {message.content}')
            translated_msg = translator.decode_sequence(message.content)
            await message.reply(translated_msg)
            function.print_detail('INFO', message.author, message.guild, message.channel, f'Send message: {translated_msg}')


async def setup(bot):
    await bot.add_cog(Events(bot))